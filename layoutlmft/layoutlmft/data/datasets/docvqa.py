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
                            names=["OTHER", "B-ANSWER", "I-ANSWER"]
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
        downloaded_file = "/scratch/DocVQA"
        #downloaded_file = "/home/rishabh.maheshwary/.cache/huggingface/datasets/downloads/extracted/f685dfeb0c20c80cff4f0c036a409562f9935ad59f7215248630987debe06560"
        #downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/data/docvqa/train/", "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/data/docvqa/val/", "split": "val"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/data/docvqa/val/", "split": "val"}
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
                ans_lower = ans.lower()
                ans_list = ans_lower.split(" ")
                all_answers.append(ans_list)
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
            min_len_ans = 1000000
            max_len_ans = 0
            for line in lines:
                bbox = line["boundingBox"]
                text = line["text"]
                words = line["words"]
                new_words = []
                answer_flag = 0
                for word in words:
                    word_bbox = word["boundingBox"]
                    word_text = word["text"]
                    tokens.append(word_text.lower())
                    #bboxes.append(word_bbox)
                    found = 0
                    bboxes.append(normalize_bboxes_docvqa([word_bbox], size[0], size[1]))
                    for ans_list in all_answers:
                        min_len_ans = min(min_len_ans,len(ans_list))
                        max_len_ans = max(max_len_ans,len(ans_list))
                        if word_text.lower() in ans_list:
                            if answer_flag == 0:
                                qa_tags.append("B-ANSWER")
                                found+=1
                                answer_flag+=1
                            else:
                                qa_tags.append("I-ANSWER")
                                found+=1
                                answer_flag+=1
                            break
                    if found == 0:
                        qa_tags.append("OTHER")
            end_idx = 0
            start_idx = 0
            cur_cnt=0
            min_cnt = -1
            start_end = []
            assert len(qa_tags) == len(tokens)

            for i in range(len(tokens)):

                if qa_tags[i] == "B-ANSWER" or qa_tags[i] == "I-ANSWER":
                    cur_cnt+=1
                else:
                    if cur_cnt > min_cnt:
                       end_idx = i - 1
                       start_idx = i - cur_cnt
                       min_cnt = cur_cnt

                    cur_cnt=0

            for i in range(len(qa_tags)):
                if i>=start_idx and i<=end_idx:
                    continue
                else:
                    qa_tags[i] = "OTHER"

            yield i, {"id": str(i), "question": question.split(" "), "tokens": tokens, "bboxes": bboxes, "ner_tags": qa_tags, "image": image, "image_path": img_path}
