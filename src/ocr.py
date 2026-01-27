import os, sys, tempfile, torch, subprocess, fitz, time, contextlib
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

class Pipeline:

	def __init__(self, batch):
		self.batch = batch
		self.mode = "Not Selected"
		self.dir = None
		self.documents = os.listdir(self.batch)
		self.num_docs = len(self.documents)
		self.num_pages = 0
		self.pipestats = {
			"S1": None,
			"S2": None,
			"S3": None,
			}

	def __str__(self):
		return f"""
			\nPipeline executed on {self.num_docs} documents for a total of {self.num_pages} pages.
			\nRuntime for {self.mode} version of DeepSeek-OCR:
				\n\tPreprocessing Step: {self.pipestats['S1']} seconds per document
				\n\tData Extraction Step: {self.pipestats['S2']} seconds per page
				\n\tPostprocessing Step: {self.pipestats['S3']} seconds per page
			"""
#|--------------------------------------------------------------|
#|		OCR Pipeline					|
#|								|
#|		1) Preprocess PDF Documents			|
#|		2) Scan documents with DeepSeek-OCR		|
#|		3) Convert results back to PDF			|
#|--------------------------------------------------------------|

	def _preprocess(self):

		process_time = 0
		dpaths = {}
		for dname in self.documents:

			start = time.time()
			dpath = os.path.join(self.batch, dname)
			name = Path(dpath).stem
			images = convert_from_path(dpath)
			img_paths = []
			self.num_pages += len(images)
			for i, img in enumerate(images,start=1):
				page_name = f'{name}_page_{i}.png'
				out_path = os.path.join(self.dir, page_name)
				img.save(out_path, 'PNG')
				img_paths.append(out_path)
			dpaths[name] = img_paths

			process_time += time.time() - start

		self.pipestats['S1'] = round((process_time / self.num_docs), 2)
		return dpaths

	def _scan(self, docs):
		with tempfile.TemporaryDirectory() as tmpdir:

			model = DeepSeek(tmpdir,self.mode)

			outdir = os.path.join(Path.cwd(), "outputs")
			os.makedirs(outdir, exist_ok=True)

			extract_time = 0
			vis_time = 0
			for dname, images in docs.items():
				out_path = os.path.join(outdir,dname + "_results.pdf")
				pdf = fitz.open()
				for img in images:
					start = time.time()
					result = model._extract(img)
					extract_time += time.time() - start

					page_path = self._convert(result)
					with fitz.open(page_path) as pg:
						pdf.insert_pdf(pg)
					vis_time += time.time() - (start + extract_time)

				pdf.save(out_path)
				pdf.close()

		self.pipestats['S2'] = round((extract_time / self.num_pages), 2)
		self.pipestats['S3'] = round((vis_time / self.num_pages), 2)

	def _convert(self, page):
		tmpdir = Path(page).parent
		tmppath = os.path.join(tmpdir, "temp.html")
		out_path = os.path.join(tmpdir, "page.pdf")
		cmd = ['pandoc', page, '-f', 'markdown_mmd+raw_html', '-t', 'html', '-o', tmppath]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		cmd = ['wkhtmltopdf', '--enable-local-file-access', tmppath, out_path]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		return out_path

	def execute(self, mode="Base"):
		with tempfile.TemporaryDirectory() as tmpdir:
			self.dir = tmpdir
			self.mode = mode
			docs = self._preprocess()
			self._scan(docs)

class DeepSeek:
	def __init__(self, outdir, mode='base'):

		self.MODES = {
			'tiny': dict(base_size=512, image_size=512, crop_mode=False),
			'small': dict(base_size=640, image_size=640, crop_mode=False),
			'base': dict(base_size=1024, image_size=1024, crop_mode=False),
			'large': dict(base_size=1280, image_size=1280, crop_mode=False),
			'gundam': dict(base_size=1024, image_size=640, crop_mode=True)
			}
		self.config = self.MODES[mode.lower()]

		self.prompt = '<image>\n<|grounding|>Convert the document to markdown.'
		self.base_size = self.config['base_size']
		self.image_size = self.config['image_size']
		self.crop_mode = self.config['crop_mode']

		self.model_dir = '.ds_ocr/models/DeepSeek-OCR'
		self.outdir = outdir

		self.device = torch.device('cuda')
		self.dtype = torch.float32

		self.tokenizer = AutoTokenizer.from_pretrained(
				self.model_dir,
				local_files_only=True,
				trust_remote_code=True
				)

		self.model = AutoModel.from_pretrained(
				self.model_dir,
				local_files_only=True,
				trust_remote_code=True,
				attn_implementation='eager',
				torch_dtype = self.dtype
				)

		self.model = self.model.to(self.device).eval()

	def _extract(self, img):

		# NOTE: DeepSeek outputs bounding box coordinates. These are only visible in stdout.
		#	Remove the "with" statements and unindent the "result" if you wish to make this output visible again.

		with open(os.devnull, 'w') as fnull:
			with contextlib.redirect_stdout(fnull):
				result = self.model.infer(
					self.tokenizer,
					prompt=self.prompt,
					image_file = img,
					output_path = self.outdir,
					base_size=self.base_size,
					image_size=self.image_size,
					crop_mode=self.crop_mode,
					save_results=True,
					test_compress=False
					)

		pred = os.path.join(self.outdir, "result.mmd")
		torch.cuda.empty_cache()

		return pred

folder = sys.argv[1]
ocr = Pipeline(folder)
ocr.execute("Base")
print(ocr)
