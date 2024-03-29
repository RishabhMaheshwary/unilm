# coding=utf-8

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bbox

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


#datasets.features.ClassLabel(
#                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
#                        )
class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    #"question": datasets.Sequence(datasets.Value("string")),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #downloaded_file = "/scratch/layoutlm/DocVQA"
        downloaded_file = "/home/rishabh.maheshwary/.cache/huggingface/datasets/downloads/extracted/f685dfeb0c20c80cff4f0c036a409562f9935ad59f7215248630987debe06560"
        #downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        breakpoint()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/", "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/", "split": "val"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/", "split": "val"}
            )
        ]

    def _generate_examples(self, filepath, split):
        #filepath = "/home/rishabh.maheshwary/LayoutLM/data/training_data/"
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        #breakpoint()
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if "text" in w and w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image, "image_path": image_path}

    ##DOCVQA preprocessing

    def _generate_examples_docvqa(self, filepath, split):
        logger.info("⏳ Generating examples from = %s", filepath)

        if split == "train":
            f_pth = os.path.join(filepath, "train_v1.0.json")
        else:
            f_pth = os.path.join(filepath, "val_v1.0.json")
        f = open(f_pth)
        data = json.load(f)
        data = data["data"]
        results = []
        cntr = 10
        for i in range(len(data)):
            cntr-=1
            if cntr == 0:
                break
            tokens, bboxes, qa_tags = [], [], []
            question = data[i]["question"]
            answers = data[i]["answers"]
            all_answers = []
            for ans in answers:
                all_answers.append(ans.lower())
            image_file = data[i]["image"]
            image_name = image_file[image_file.find("/")+1:]
            img_path = os.path.join(filepath, image_file)
            image, size = load_image(img_path)
            ocr_file = image_name[:-4]
            ocr_file +=".json"
            ocr_file  = "ocr_results/"+ocr_file
            ocr_f = os.path.join(filepath, ocr_file)
            ocr_f = open(ocr_f)
            ocr_data = json.load(ocr_f)
            ocr_data = ocr_data["recognitionResults"][0]
            lines = ocr_data["lines"]
            for line in lines:
                bbox = line["boundingBox"]
                text = line["text"]
                words = line["words"]
                new_words = []
                for word in words:
                    word_bbox = word["boundingBox"]
                    word_text = word["text"]
                    tokens.append(word_text.lower())
                    x1, y1, x2, y2, x3, y3, x4, y4 = word_bbox
                    bboxes.append(normalize_bbox([x1, y1, x3, y3],size))
                    if word_text.lower() in all_answers:
                        qa_tags.append("ANSWER")
                    else:
                        qa_tags.append("OTHER")

            yield i, {"id": str(i), "question": question.split(" "), "tokens": tokens, "bboxes": bboxes, "ner_tags": qa_tags, "image": image, "image_path": img_path}
