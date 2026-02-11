import os, sys
import threading, psutil, statistics
import tempfile, torch, subprocess, fitz, time, shutil
from pdf2image import convert_from_path
from pathlib import Path
from nemotron_ocr.inference.pipeline import NemotronOCR
from PIL import Image, ImageOps

class Memory:
	def __init__(self, interval=0.05):
		self.interval = interval
		self._stop = threading.Event()
		self.samples = []
		self.proc = psutil.Process(os.getpid())
		self._thread = None

	def _sample(self):
		rss = self.proc.memory_info().rss
		self.samples.append(rss)

	def _run(self):
		self._sample()
		while not self._stop.wait(self.interval):
			self._sample()

	def __enter__(self):
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()
		return self

	def __exit__(self, exctype, exc, tb):
		self._stop.set()
		self._thread.join()

	def summary(self):
		rss = [(s / (1024**2)) for s in self.samples]
		rss_mean = round(statistics.fmean(rss), 2)
		rss_peak = round(max(rss), 2)
		return {
			"rss_mean": rss_mean,
			"rss_peak": rss_peak,
			}
class Pipeline:

	def __init__(self, batch):
		self.batch = batch
		self.dir = None
		self.documents = os.listdir(self.batch)
		self.num_docs = len(self.documents)
		self.num_pages = 0
		self.DPI = 300 # Optimize this parameter
		self.runstats = {
			"S1": None,
			"S2": None,
			"S3": None,
			}
		self.memstats = {
			"rss_mean": None,
			"rss_peak": None,
			"gpu_peak": None,
			}

		self.global_rss_peak = 0

	def __str__(self):
		return f"""
			\nPipeline executed on {self.num_docs} documents for a total of {self.num_pages} pages.
			\nRuntime for NemotronOCR:
				\n\tPreprocessing Step: {self.runstats['S1']} seconds per document
				\n\tData Extraction Step: {self.runstats['S2']} seconds per page
				\n\tPostprocessing Step: {self.runstats['S3']} seconds per page
			\nMemory Characteristics:
				\n\tAverage RAM used per page: {self.memstats['rss_mean']}MiB
				\n\tPeak RAM usage: {self.memstats['rss_peak']}MiB
				\n\tPeak GPU usage: {self.memstats['gpu_peak']}MiB
			"""
#|--------------------------------------------------------------|
#|		OCR Pipeline					|
#|								|
#|		1) Preprocess PDF Documents			|
#|		2) Scan documents with OCR			|
#|		3) Convert results back to PDF			|
#|--------------------------------------------------------------|

	def _preprocess(self):

		process_time = 0
		dpaths = {}
		for dname in self.documents:

			start = time.time()
			dpath = os.path.join(self.batch, dname)
			name = Path(dpath).stem
			images = convert_from_path(dpath, dpi=self.DPI, use_cropbox=True)
			img_paths = []
			self.num_pages += len(images)
			for i, img in enumerate(images,start=1):
				page_name = f'{name}_page_{i}.png'
				out_path = os.path.join(self.dir, page_name)
				img = ImageOps.exif_transpose(img) # Standardize page orientation
				if img.mode != "RGB":
					img = img.convert("RGB")
				img.save(out_path, format='PNG')
				img_paths.append(out_path)
			dpaths[name] = img_paths

			process_time += time.time() - start

		self.runstats['S1'] = round((process_time / self.num_docs), 2)
		return dpaths

	def _scan(self, docs):
		with tempfile.TemporaryDirectory() as tmpdir:

			model = OCR(tmpdir)

			outdir = os.path.join(Path.cwd(), "outputs")
			os.makedirs(outdir, exist_ok=True)

			extract_time = 0
			vis_time = 0
			rss_mean = 0
			rss_peak = 0
			gpu_peak = 0

			for dname, images in docs.items():
				out_path = os.path.join(outdir,dname + "_results.pdf")
				pdf = fitz.open()
				for i, img in enumerate(images, start=1):
					raw_out = os.path.join(outdir,dname + f"_raw_result_page{i}.txt")
					raw_err = os.path.join(outdir,dname + f"_err_log_page{i}.txt")
					torch.cuda.reset_peak_memory_stats()
					start = time.time()
					with Memory() as mem:
						result = model._extract(img)
					extract_time += time.time() - start
					torch.cuda.synchronize()

					sums = mem.summary()
					gpu_peak = max(gpu_peak, round(torch.cuda.max_memory_allocated() / (1024**2), 2))
					rss_mean += sums['rss_mean']
					rss_peak = max(rss_peak, sums['rss_peak'])

					start = time.time()
					page_path = self._convert(result, i)
					with fitz.open(page_path) as pg:
						pdf.insert_pdf(pg)

					shutil.move(os.path.join(Path(result).parent,"raw_output.txt"), raw_out)
					shutil.move(os.path.join(Path(result).parent,"err_log.txt"), raw_err)
					vis_time += time.time() - start

				pdf.save(out_path)
				pdf.close()

		self.runstats['S2'] = round((extract_time / self.num_pages), 2)
		self.runstats['S3'] = round((vis_time / self.num_pages), 2)
		self.memstats['rss_mean'] = round(rss_mean / self.num_pages, 2)
		self.memstats['rss_peak'] = round(rss_peak, 2)
		self.memstats['gpu_peak'] = round(gpu_peak, 2)

	def _convert(self, page, page_num):
		tmpdir = Path(page).parent
		tmppath = os.path.join(tmpdir, "temp.html")
		out_path = os.path.join(tmpdir, f"page_{page_num}.pdf")
		cmd = ['pandoc', page, '-f', 'markdown_mmd+raw_html', '-t', 'html', '-o', tmppath]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		cmd = ['wkhtmltopdf', '--enable-local-file-access', tmppath, out_path]
		res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		return out_path

	def execute(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			self.dir = tmpdir
			docs = self._preprocess()
			self._scan(docs)

class OCR:

	def __init__(self, tmpdir):
		self.model = NemotronOCR()
		self.dir = tmpdir
		self.merge_level="paragraph" # Determine which groups results the best
		self.visualize=False # Set to True and incorporate visualizaiton later

	def _extract(self, img):
		results = self.model(img, merge_level=self.merge_level, visualize=self.visualize)
		print(results)
		sys.exit()
		return results

folder = sys.argv[1]
pipe = Pipeline(folder)
pipe.execute()
