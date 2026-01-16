import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import time, fitz, pytesseract, re, tempfile, torch, io, sys, html
import numpy as np
from Levenshtein import distance
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from PIL import Image, ImageOps
from IOManager import Manager
from paddleocr import PaddleOCR

class Processor:
	def __init__(self, scale=2.0):
		self._io = Manager()
		self.samples = self._io.samples
		self._dir = self._io._indir
		self.scale = scale
		self.images = self._process()

	def _process(self):
		print("Processing PDF files to Images...")
		group_pages = {}
		for group, docs in self.samples.items():
			print(f"Processing {group}...")
			group_dir = self._dir / group
			doc_pages = {}
			for doc in docs:
				pdf_path = group_dir / doc
				pdf = fitz.open(pdf_path)

				page_results = []

				for pagenum, page in enumerate(pdf, start=1):
					text = page.get_text('text')
					render = fitz.Matrix(self.scale, self.scale)
					pix = page.get_pixmap(matrix=render, alpha=False, dpi=600)
					img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
					page_results.append({
							"page_num": pagenum,
							"image": img,
							"gt_text": text,
							"width": pix.width,
							"height": pix.height,
							"scale": self.scale
						})
				pdf.close()
				doc_pages[doc] = page_results

			group_pages[group] = doc_pages

		print("Processing complete.\n")

		return group_pages

	def save(self, predictions, model):
		self._io.save_output(predictions, model)

class Reader:
	def __init__(self, data,model):
		self.images = data.images
		self._dir = data._dir
		self._model = model.lower()

	def read(self):
		group_results = {}
		total_pages = 0

		start = time.perf_counter()

		print("Selected Model: ", self._model)
		for group, docs in self.images.items():
			print(f"Running OCR on {group}'s documents")
			doc_results = {}
			for name,pages in docs.items():
				total_pages += len(pages)

				dpages = [p['image'] for p in pages]

				match self._model:
					case 'tesseract':
						output = Tesseract(dpages).extract()

					case 'deepseek':
						output = DeepSeek(dpages, 'large').extract()

					case 'paddle':
						output = Paddle(dpages).extract()

				doc_results[name] = output

			group_results[group] = doc_results
		runtime = time.perf_counter() - start

		print(f"Elapsed time: {runtime:.4f} seconds")
		print(f"Number of pages read: {total_pages}")
		print(f"Average reading speed: {(runtime / total_pages):.4f} seconds per page")

		return group_results

class DeepSeek:
	def __init__(self, dpages, mode):
		self.dpages = dpages
		self.mode = mode.lower()
		self.model_dir = Path('./models/DeepSeek-OCR')
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True, local_files_only=True)
		self.model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True, local_files_only=True, use_safetensors=True, _attn_implementation='eager').eval()
		if self.device == 'cuda':
			self.model = self.model.cuda().to(torch.bfloat16)
		self.task = "ocr"
		self.prompt = "<image>\n<|grounding|>Convert the document to markdown."
#		self.prompt = "<image>\nFree OCR."
		self.MODES = {
			'tiny': dict(base_size=512, image_size=512, crop_mode=False),
			'small': dict(base_size=640, image_size=640, crop_mode=False),
			'base': dict(base_size=1024, image_size=1024, crop_mode=False),
			'large': dict(base_size=1280, image_size=1280, crop_mode=False),
			'gundam': dict(base_size=1024, image_size=640, crop_mode=True)
			}
		m = self.MODES[self.mode]
		self.base_size = m['base_size']
		self.image_size = m['image_size']
		self.crop_mode = m['crop_mode']
		self.max_new_tokens = 2048
		self.temperature = 0.0
		self.do_sample = False

	def extract(self):
		final_text = []

		with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmpdir:
			for idx, page in enumerate(self.dpages, start=1):
				img = page.convert('RGB') # Ensure image is RGB
				img_path = os.path.join(tmpdir, f"deepseek_ocr_{idx:05d}.png")
				img.save(img_path, format="PNG")

				revert_stdout = sys.stdout
				buffer = io.StringIO()
				sys.stdout = buffer

				outdir = os.path.join(os.path.dirname(__file__), 'ocr-outputs','deepseek-images')
				res = self.model.infer(self.tokenizer,
						prompt=self.prompt,
						image_file=img_path,
						output_path=outdir,
						base_size=self.base_size,
						image_size=self.image_size,
						crop_mode=self.crop_mode,
						save_results=True,
						test_compress=False
						)

				sys.stdout = revert_stdout
				raw_text = buffer.getvalue()

				text = self._remove_grounding(raw_text)
				clean_text = self._format_output(text)
				final_text.append(clean_text)

		return final_text

	def _remove_grounding(self, text):
		s = html.unescape(text).replace("<$/", "</")
		s = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", s, flags=re.S)
		s = re.sub(r"<\|det\|>.*?<\|/det\|>", "", s, flags=re.S)
		return s.strip()

	def _format_output(self, text):
		def _create_table(m):
			tbl = m.group(0)
			rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", tbl, flags=re.S|re.I)
			lines = []
			for row in rows:
				cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, flags=re.S|re.I)
				cells = [html.unescape(c).strip() for c in cells]
				if cells:
					lines.append(" ".join(cells))
			return "\n".join(lines)

		banner = re.compile(
			r"""
			^\s*={5,}\s*\n
			\s*BASE:\s*torch\.Size\([^\n]*\)\s*\n
			\s*PATCHES:\s*torch\.Size\([^\n]*\)\s*\n
			^\s*={5,}\s*$
			""",
			flags=re.MULTILINE | re.IGNORECASE | re.VERBOSE,
			)

		# Normalize new lines
		text = text.replace("\r\n", "\n").replace("\r", "\n")

		# Remove banner block
		text = banner.sub("", text)

		# Remove leftover banner artifacts
		text = re.sub(r"^\s*BASE:\s*torch\.Size\([^\n]*\)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
		text = re.sub(r"^\s*PATCHES:\s*torch\.Size\([^\n]*\)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
		text = re.sub(r"^\s*=+\s*$", "", text, flags=re.MULTILINE)

		# Remove grounding
		text = self._remove_grounding(text)

		out = re.sub(r"<table\b[^>]*>.*?</table>", _create_table, text, flags=re.S | re.I)
		out = re.sub(r"</?[^>\n]+>", "", out)

		# Normalize whitespace
		out = re.sub(r"[ \t]+", " ", out)
		out = re.sub(r"\s*\n\s*", "\n", out)
		out = re.sub(r"\n{2,}", "\n", out).strip()

		return out
