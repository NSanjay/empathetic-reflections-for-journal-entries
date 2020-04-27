"""Preprocess csv files to generate train / val /test splits"""

__author__ = "NSanjay"
from pathlib import *
import csv
import json
import re


class DataPreprocessor:
    def __init__(self, data_dir: str, file_path: str) -> None:
        self.data_dir = data_dir
        self.file_path = file_path


    def _preprocess_field(self, field_text: str) -> str:
        field_text = field_text.replace("\n", " ")
        field_text = re.sub(r'[\s]+', " ", field_text)
        return field_text

    def prepare_input_files(self) -> None:
        with open(Path(self.data_dir, self.file_path), "r", encoding='utf-8-sig') as in_file, \
                open(Path(self.data_dir, "train.jsonl"), "w+",encoding='utf-8') as train_file, \
                open(Path(self.data_dir, "val.jsonl"), "w+",encoding='utf-8') as val_file, \
                open(Path(self.data_dir, "test.jsonl"), "w+",encoding='utf-8') as test_file:
            csv_reader = csv.DictReader(in_file, quoting=csv.QUOTE_MINIMAL)
            val_questions = set()
            test_questions = set()
            for i, row in enumerate(csv_reader):
                a_json = {}
                a_json["example_id"] = row["split"] + "-" + str(i)
                if row["questionText"] == "" or row["answerText"] == "":
                    continue
                a_json["question_text"] = self._preprocess_field(row["questionText"])

                a_json["answer_text"] = self._preprocess_field(row["answerText"])
                period_split = a_json["answer_text"].split(".")
                a_json["answer_text"] = ". ".join(period_split[:2])

                out_json = json.dumps(a_json, ensure_ascii=False)
                if row["split"] == "train":
                    train_file.write(out_json+"\n")

                elif row["split"] == "val":
                    if a_json["question_text"] not in val_questions:
                        val_file.write(out_json+"\n")
                        val_questions.add(a_json["question_text"])

                elif row["split"] == "test":
                    if a_json["question_text"] not in test_questions:
                        test_file.write(out_json+"\n")
                        test_questions.add(a_json["question_text"])

                else:
                    print(row["split"])
                    raise Exception("unexpected value")


if __name__ == "__main__":
    preprocessor = DataPreprocessor("../data", "counsel_chat.csv")
    preprocessor.prepare_input_files()
