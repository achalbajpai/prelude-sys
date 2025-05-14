import os
import cv2
import numpy as np
import pandas as pd
import PyPDF2
import re
import spacy
from pdf2image import convert_from_path
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


class MedicalRecordPageClustering:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.page_texts = []
        self.page_features = {}
        self.page_clusters = {}
        self.nlp = spacy.load("en_core_web_sm")

    def extract_text_from_pdf(self):
        print("Extracting text from PDF...")

        pdf_reader = PyPDF2.PdfReader(self.pdf_path)
        num_pages = len(pdf_reader.pages)

        for page_num in tqdm(range(num_pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            if len(text.strip()) < 100:
                try:
                    images = convert_from_path(
                        self.pdf_path, first_page=page_num + 1, last_page=page_num + 1
                    )
                    text = pytesseract.image_to_string(images[0])
                except Exception as e:
                    print(
                        f"Warning: OCR failed for page {page_num + 1}. Using partial text. Error: {e}"
                    )

            self.page_texts.append({"page_num": page_num + 1, "text": text})

        print(f"Extracted text from {len(self.page_texts)} pages")
        return self.page_texts

    def extract_medical_entities(self, text):
        doc = self.nlp(text)

        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        dates = re.findall(date_pattern, text)

        provider_pattern = (
            r"Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+|[A-Z][a-z]+\s+[A-Z][a-z]+,\s+M\.?D\.?"
        )
        providers = re.findall(provider_pattern, text)

        facility_pattern = r"(Hospital|Medical Center|Clinic|Care|Health|Center)"
        facilities = [
            sent.text for sent in doc.sents if re.search(facility_pattern, sent.text)
        ]

        lines = text.split("\n")
        headers = []
        for i in range(min(5, len(lines))):
            if len(lines[i].strip()) > 3 and len(lines[i].strip()) < 50:
                headers.append(lines[i].strip())

        return {
            "dates": dates,
            "providers": providers,
            "facilities": facilities,
            "headers": headers,
        }

    def extract_page_features(self):
        print("Extracting features from pages...")

        for page_data in tqdm(self.page_texts):
            page_num = page_data["page_num"]
            text = page_data["text"]

            entities = self.extract_medical_entities(text)
            doc_type = self.identify_document_type(text)
            structure_features = self.extract_structure_features(text)

            self.page_features[page_num] = {
                "text": text,
                "entities": entities,
                "doc_type": doc_type,
                "structure_features": structure_features,
            }

        return self.page_features

    def identify_document_type(self, text):
        text = text.lower()

        doc_types = {
            "lab": [
                "laboratory",
                "lab report",
                "test results",
                "chemistry",
                "hematology",
            ],
            "progress": [
                "progress note",
                "office visit",
                "progress report",
                "clinical note",
            ],
            "discharge": [
                "discharge summary",
                "discharge plan",
                "discharge instructions",
            ],
            "radiology": ["radiology", "x-ray", "mri", "ct scan", "ultrasound"],
            "prescription": ["prescription", "medication", "pharmacy", "rx"],
            "consultation": ["consultation", "referral", "consult"],
        }

        for doc_type, keywords in doc_types.items():
            for keyword in keywords:
                if keyword in text:
                    return doc_type

        return "unknown"

    def extract_structure_features(self, text):
        lines = text.split("\n")

        table_line_count = 0
        for line in lines:
            if len(line.strip()) > 0 and "  " in line:
                table_line_count += 1

        has_footer = False
        if len(lines) > 5:
            footer_text = "\n".join(lines[-3:])
            if re.search(r"page\s+\d+|\d+\s+of\s+\d+", footer_text.lower()):
                has_footer = True

        page_numbers = re.findall(r"page\s+(\d+)|\s+(\d+)\s+of\s+\d+", text.lower())

        return {
            "line_count": len(lines),
            "avg_line_length": (
                np.mean([len(line) for line in lines if len(line) > 0]) if lines else 0
            ),
            "table_line_count": table_line_count,
            "has_footer": has_footer,
            "page_numbers": page_numbers,
        }

    def compute_page_similarities(self):
        print("Computing page similarities...")

        page_nums = sorted(list(self.page_features.keys()))
        n_pages = len(page_nums)

        similarity_matrix = np.zeros((n_pages, n_pages))

        texts = [self.page_features[p]["text"] for p in page_nums]

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        text_similarities = cosine_similarity(tfidf_matrix)

        for i in range(n_pages):
            for j in range(n_pages):
                pi = page_nums[i]
                pj = page_nums[j]

                text_sim = text_similarities[i, j]

                doc_type_sim = (
                    1.0
                    if self.page_features[pi]["doc_type"]
                    == self.page_features[pj]["doc_type"]
                    else 0.0
                )

                header_sim = self.compute_header_similarity(
                    self.page_features[pi]["entities"]["headers"],
                    self.page_features[pj]["entities"]["headers"],
                )

                page_continuity = 1.0 if abs(pi - pj) == 1 else 0.0

                similarity_matrix[i, j] = (
                    0.5 * text_sim
                    + 0.2 * doc_type_sim
                    + 0.2 * header_sim
                    + 0.1 * page_continuity
                )

        return similarity_matrix, page_nums

    def compute_header_similarity(self, headers1, headers2):
        if not headers1 or not headers2:
            return 0.0

        max_sim = 0.0
        for h1 in headers1:
            for h2 in headers2:
                words1 = set(h1.lower().split())
                words2 = set(h2.lower().split())

                if not words1 or not words2:
                    continue

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                sim = intersection / union if union > 0 else 0
                max_sim = max(max_sim, sim)

        return max_sim

    def cluster_pages(self, threshold=0.6):
        print("Clustering pages...")

        similarity_matrix, page_nums = self.compute_page_similarities()
        n_pages = len(page_nums)

        clusters = {}
        cluster_id = 1
        assigned_pages = set()

        for i in range(n_pages):
            if page_nums[i] in assigned_pages:
                continue

            current_cluster = [page_nums[i]]
            assigned_pages.add(page_nums[i])

            j = i
            while j < n_pages - 1:
                if similarity_matrix[j, j + 1] >= threshold:
                    current_cluster.append(page_nums[j + 1])
                    assigned_pages.add(page_nums[j + 1])
                    j += 1
                else:
                    break

            clusters[cluster_id] = current_cluster
            cluster_id += 1

        self.page_clusters = clusters
        return clusters

    def extract_record_metadata(self, cluster):
        metadata = {
            "dos": None,
            "provider": None,
            "facility": None,
            "header": None,
            "category": None,
        }

        all_dates = []
        all_providers = []
        all_facilities = []
        all_headers = []

        for page_num in cluster:
            page_data = self.page_features[page_num]

            all_dates.extend(page_data["entities"]["dates"])
            all_providers.extend(page_data["entities"]["providers"])
            all_facilities.extend(page_data["entities"]["facilities"])
            all_headers.extend(page_data["entities"]["headers"])

        if all_dates:
            metadata["dos"] = max(set(all_dates), key=all_dates.count)

        if all_providers:
            metadata["provider"] = max(set(all_providers), key=all_providers.count)

        if all_facilities:
            metadata["facility"] = max(set(all_facilities), key=all_facilities.count)

        if all_headers:
            metadata["header"] = max(set(all_headers), key=all_headers.count)

        doc_types = [self.page_features[page_num]["doc_type"] for page_num in cluster]
        if doc_types:
            most_common_type = max(set(doc_types), key=doc_types.count)

            category_map = {
                "lab": 24,
                "progress": 16,
                "discharge": 10,
                "radiology": 30,
                "prescription": 20,
                "consultation": 15,
                "unknown": 0,
            }

            metadata["category"] = category_map.get(most_common_type, 0)

        return metadata

    def generate_output_csv(self, output_path):
        print("Generating output CSV...")

        results = []

        for cluster_id, pages in self.page_clusters.items():
            metadata = self.extract_record_metadata(pages)

            parent_key = f"12099{cluster_id}"

            for i, page_num in enumerate(pages):
                result = {
                    "pagenumber": page_num,
                    "category": metadata["category"],
                    "isreviewable": "TRUE",
                    "dos": metadata["dos"],
                    "provider": (
                        f"{metadata['provider']} - {metadata['facility']}"
                        if metadata["provider"] and metadata["facility"]
                        else ""
                    ),
                    "referencekey": f"{parent_key}{i}",
                    "parentkey": parent_key if i > 0 else 0,
                    "lockstatus": "L",
                    "header": metadata["header"],
                    "facilitygroup": "",
                    "reviewerid": 287,
                    "qcreviewerid": 322,
                    "isduplicate": "FALSE",
                }
                results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Output CSV saved to {output_path}")
        return df

    def visualize_clusters(self, output_path):
        print("Visualizing page clusters...")

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.page_clusters)))

        plt.figure(figsize=(12, 8))

        for i, (cluster_id, pages) in enumerate(self.page_clusters.items()):
            plt.scatter(
                pages,
                [i] * len(pages),
                c=[colors[i]],
                label=f"Cluster {cluster_id}",
                s=100,
            )

            plt.plot(pages, [i] * len(pages), c=colors[i], alpha=0.5)

        plt.yticks(
            range(len(self.page_clusters)),
            [f"Cluster {cid}" for cid in self.page_clusters.keys()],
        )
        plt.xlabel("Page Number")
        plt.ylabel("Cluster")
        plt.title("Medical Record Page Clusters")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(output_path)
        print(f"Cluster visualization saved to {output_path}")

    def process(self, output_csv_path, visualization_path=None):
        self.extract_text_from_pdf()
        self.extract_page_features()
        self.cluster_pages()

        df = self.generate_output_csv(output_csv_path)

        if visualization_path:
            self.visualize_clusters(visualization_path)

        return df


if __name__ == "__main__":
    pdf_path = "Sample Document.pdf"
    output_csv = "Clustered_Medical_Records.csv"
    visualization = "Page_Clusters_Visualization.png"

    clustering = MedicalRecordPageClustering(pdf_path)
    result_df = clustering.process(output_csv, visualization)

    print("Page clustering complete!")
    print(f"Found {len(clustering.page_clusters)} distinct medical record groups")
    for cluster_id, pages in clustering.page_clusters.items():
        metadata = clustering.extract_record_metadata(pages)
        print(
            f"Cluster {cluster_id}: Pages {pages} - {metadata['header']} ({metadata['dos']})"
        )
