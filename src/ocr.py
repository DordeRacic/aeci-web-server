import os, sys, tempfile, torch, shutil, io, subprocess
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

class Pipeline:

	def __init__(self, batch):
		self.batch = batch
		self.documents = os.listdir(self.batch)

#|--------------------------------------------------------------|
#|		OCR Pipeline					|
#|								|
#|		1) Preprocess PDF Documents			|
#|		2) Scan documents with DeepSeek-OCR		|
#|		3) Convert results back to PDF			|
#|--------------------------------------------------------------|
	def _preprocess(self):

		for dname in self.documents:

			dpath = os.path.join(self.batch, dname)
			name = Path(dpath).stem
			images = convert_from_path(dpath)
			img_paths = []

			for i, img in enumerate(images,start=1):
				page_name = f'{name}_page_{i}.png'
				out_path = os.path.join(self.dir, page_name)
				img.save(out_path, 'PNG')
				img_paths.append(out_path)
		return img_paths

	def _scan(self, images):
		with tempfile.TemporaryDirectory() as tmpdir:

			model = DeepSeek(tmpdir)

			outdir = os.path.join(Path.cwd(), "outputs")
			for img in images:
				result = model._extract(img)

				dname = Path(img).stem
				self._convert(result, dname, outdir)

	def _convert(self, page, name, outdir):
		out_path = os.path.join(outdir, name + "_results.pdf")
		tmpdir = Path(page).parent
		tmppath = os.path.join(tmpdir, "temp.html")
		cmd = ['pandoc', page, '-f', 'markdown_mmd+raw_html', '-t', 'html', '-o', tmppath]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		cmd = ['wkhtmltopdf', '--enable-local-file-access', tmppath, out_path]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


	def execute(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			self.dir = tmpdir
			images = self._preprocess()
			self._scan(images)

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
Pipeline(folder).execute()
