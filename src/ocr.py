import os, sys, tempfile, torch
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer

class Pipeline:

	def __init__(self, batch):
		self.batch = batch
		self.documents = os.listdir(self.batch)
		self.model = DeepSeek()

#|--------------------------------------------------------------|
#|		OCR Pipeline					|
#|								|
#|		1) Preprocess PDF Documents			|
#|		2) Scan documents with DeepSeek-OCR		|
#|--------------------------------------------------------------|
	def _preprocess(self):

		for dname in self.documents:

			dpath = os.path.join(self.batch, dname)
			images = convert_from_path(dpath)
			img_paths = []

			for i, img in enumerate(images):
				page_name = f'{dname}_page_{i}.png'
				out_path = os.path.join(self.dir, page_name)
				img.save(out_path, 'PNG')
				img_paths.append(out_path)
		return img_paths

	def _scan(self, images):
		model = DeepSeek()
		model._extract(images)

	def execute(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			self.dir = tmpdir
			images = self._preprocess()
			self._scan(images)

class DeepSeek:
	def __init__(self, mode='large'):

		self.MODES = {
			'tiny': dict(base_size=512, image_size=512, crop_mode=False),
			'small': dict(base_size=640, image_size=640, crop_mode=False),
			'base': dict(base_size=1024, image_size=1024, crop_mode=False),
			'large': dict(base_size=1280, image_size=1280, crop_mode=False),
			'gundam': dict(base_size=1024, image_size=640, crop_mode=False)
			}
		self.config = self.MODES[mode.lower()]

		self.device = 'cuda'
		self.task = 'ocr'
		self.prompt = '<image>\n<|grounding|>Convert the document to markdown.'
		self.size = self.config['base_size']
		self.img_size = self.config['image_size']
		self.crop = self.config['crop_mode']
		self.max_new_tokens = 2048
		self.temperature = 0.0
		self.do_samples = False

		self.model_dir = '.ds_ocr/models/DeepSeek-OCR'

		self.tokenizer = AutoTokenizer.from_pretrained(
				self.model_dir,
				local_files_only=True,
				trust_remote_code=True
				)

		self.model = AutoModel.from_pretrained(
				self.model_dir,
				local_files_only=True,
				trust_remote_code=True,
				attn_implementation='flash_attention_2',
				torch_dtype=torch.float16
				)
		self.model = self.model.to(self.device).eval()

	def _extract(self, images):
		with tempfile.TemporaryDirectory(prefix='deepseek_ocr_') as tmpdir:
			for idx, img in enumerate(images, start=1):
				img_path = os.path.join(tmpdir, f"deepseek_ocr_{idx:05d}.png")

folder = sys.argv[1]
Pipeline(folder).execute()
