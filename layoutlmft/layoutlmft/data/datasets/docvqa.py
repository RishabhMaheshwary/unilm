# coding=utf-8

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bboxes_docvqa

logger = datasets.logging.get_logger(__name__)



class DocvqaConfig(datasets.BuilderConfig):
    """BuilderConfig for Docvqa"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocvqaConfig, self).__init__(**kwargs)


class Docvqa(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DocvqaConfig(name="docvqa", version=datasets.Version("1.0.0"), description="DocVQA dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="DocVQA",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Sequence(datasets.Value("string")),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["OTHER", "ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation="DocVQA",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = "/scratch/layoutlm/DocVQA"
        #downloaded_file = "/home/rishabh.maheshwary/.cache/huggingface/datasets/downloads/extracted/f685dfeb0c20c80cff4f0c036a409562f9935ad59f7215248630987debe06560"
        #downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
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

    ##DOCVQA preprocessing

    def _generate_examples(self, filepath, split):
        logger.info("⏳ Generating examples from = %s", filepath)

        if split == "train":
            f_pth = os.path.join(filepath, "train_v1.0.json")
        else:
            f_pth = os.path.join(filepath, "val_v1.0.json")
        f = open(f_pth)
        data = json.load(f)
        data = data["data"]
        results = []
        for i in range(len(data)):
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
                    #bboxes.append(word_bbox)
                    bboxes.append(normalize_bboxes_docvqa([word_bbox], size[0], size[1]))
                    if word_text.lower() in all_answers:
                        qa_tags.append("ANSWER")
                    else:
                        qa_tags.append("OTHER")

            yield i, {"id": str(i), "question": question.split(" "), "tokens": tokens, "bboxes": bboxes, "ner_tags": qa_tags, "image": image, "image_path": img_path}
