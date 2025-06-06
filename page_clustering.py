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

        best_header_candidate = None
        candidate_priority = 0  # Higher is better

        # Define prioritized keywords/patterns
        # Priority 1: Exact, critical section titles (case-insensitive start)
        p1_keywords = [
            "labs",
            "progress notes",
            "medications",
            "allergies",
            "chief complaint",
            "assessment",
            "plan",
            "patient instructions",
            "discharge summary",
            "operative report",
            "consultation report",
            "history and physical",
            "radiology report",
            "pathology report",
            "operative note",
        ]
        # Priority 2: ALL CAPS general titles with common report/note words
        p2_keywords = [
            "REPORT",
            "NOTE",
            "SUMMARY",
            "RESULTS",
            "EXAMINATION",
            "STUDY",
            "FINDINGS",
            "ASSESSMENT",
            "IMPRESSION",
        ]
        # Priority 3: Title Case general titles (less emphasis than P2)
        p3_keywords = [kw.lower() for kw in p2_keywords]
        # Priority 4: "(continued)" lines (only if their base matches a known good title, handled by similarity later)
        # For now, we just want to identify if it *is* a continued line to give it lower priority than specific titles found on same page.

        for i in range(min(20, len(lines))):  # Scan first 20 lines
            line_text = lines[i].strip()
            if not line_text or len(line_text) > 100:  # Skip empty or very long lines
                continue

            line_lower = line_text.lower()
            words_in_line = line_text.split()
            num_words = len(words_in_line)

            # Check Priority 1 (Most Specific)
            for kw in p1_keywords:
                if (
                    line_lower.startswith(kw) and num_words <= 5
                ):  # Starts with specific title, short
                    if candidate_priority < 5:
                        best_header_candidate = line_text
                        candidate_priority = 5
                    break  # Found P1 match for this line
            if candidate_priority == 5:
                continue  # Move to next line if P1 found for this line

            # Check Priority 2 (ALL CAPS General Title)
            if line_text.isupper() and 1 <= num_words <= 5:
                for kw_caps in p2_keywords:
                    if (
                        kw_caps in line_text
                    ):  # No need for line_text.upper() as it's already upper
                        if candidate_priority < 4:
                            best_header_candidate = line_text
                            candidate_priority = 4
                        break
            if candidate_priority == 4:
                continue

            # Check Priority 3 (Title Case General Title)
            # Check if first word is capitalized (simple title case check)
            if (
                num_words > 0
                and words_in_line[0][0].isupper()
                and not line_text.isupper()
                and 1 <= num_words <= 6
            ):
                for kw_title in p3_keywords:
                    if kw_title in line_lower:
                        if candidate_priority < 3:
                            best_header_candidate = line_text
                            candidate_priority = 3
                        break
            if candidate_priority == 3:
                continue

            # Check Priority 4 (Continued lines - identified to be de-prioritized if other candidates exist)
            if "(continued)" in line_lower and num_words <= 5:
                if candidate_priority < 2:
                    best_header_candidate = line_text
                    candidate_priority = 2
                # Do not break or continue, a more specific title might follow on a later line in the scan range

        extracted_headers = []
        if best_header_candidate:
            extracted_headers.append(best_header_candidate)
        else:  # Fallback to the very first non-empty, reasonably short line
            for line_content in lines[
                : min(5, len(lines))
            ]:  # Check first 5 lines for any content
                cleaned_line = line_content.strip()
                if cleaned_line and len(cleaned_line) < 70:  # Reasonably short
                    extracted_headers.append(cleaned_line)
                    break
            if not extracted_headers:  # If truly nothing useful found
                extracted_headers.append("Unknown Header")

        return {
            "dates": dates,
            "providers": providers,
            "facilities": facilities,
            "headers": extracted_headers,
        }

    def extract_page_features(self):
        print("Extracting features from pages...")

        for page_data in tqdm(self.page_texts):
            page_num = page_data["page_num"]
            text = page_data["text"]

            entities = self.extract_medical_entities(text)
            # Use the primary extracted header for document type identification if available
            primary_header = entities["headers"][0] if entities["headers"] else ""
            doc_type = self.identify_document_type(
                text, primary_header
            )  # Pass both text and header
            structure_features = self.extract_structure_features(text)

            self.page_features[page_num] = {
                "text": text,
                "entities": entities,
                "doc_type": doc_type,
                "structure_features": structure_features,
            }

        return self.page_features

    def identify_document_type(self, text, primary_header=""):
        text_to_check = text.lower()

        doc_types = {
            "lab": [
                "laboratory",
                "lab report",
                "lab results",
                "test results",
                "chemistry",
                "hematology",
                "pathology report",
                "blood work",
                "report of laboratory examination",
                "laboratory findings",
            ],
            "progress": [
                "progress note",
                "office visit",
                "progress report",
                "clinical note",
                "physician's notes",
                "physician's progress note",
                "clinic visit",
                "follow-up note",
                "soap note",
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

        first_few_lines_text = text_to_check[:150]

        for doc_type, keywords in doc_types.items():
            for keyword in keywords:
                if keyword in first_few_lines_text:
                    if (
                        len(keyword.split()) > 1
                        and keyword in first_few_lines_text.split("\n")[0]
                    ):
                        return doc_type
                    if keyword in first_few_lines_text:
                        return doc_type

        for doc_type, keywords in doc_types.items():
            for keyword in keywords:
                if keyword in text_to_check:
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

                doc_type_pi = self.page_features[pi]["doc_type"]
                doc_type_pj = self.page_features[pj]["doc_type"]

                if doc_type_pi == doc_type_pj:
                    if doc_type_pi == "unknown":
                        doc_type_sim = 0.5  # Reduced similarity for two 'unknown' types
                    else:
                        doc_type_sim = (
                            1.0  # Strong similarity for two matching known types
                        )
                else:
                    doc_type_sim = 0.0  # No similarity for different types

                header_sim = self.compute_header_similarity(
                    self.page_features[pi]["entities"]["headers"],
                    self.page_features[pj]["entities"]["headers"],
                )

                # Consider similarity only for adjacent pages for the clustering logic that iterates sequentially
                # However, the full matrix might be useful for other clustering approaches. For now, focus on pi == pj or abs(pi-pj)==1
                # The current clustering logic in cluster_pages only uses similarity_matrix[j, j + 1]
                # So, this full matrix calculation is more general than strictly needed by cluster_pages, but good for potential future use.

                page_continuity = 0.0
                if abs(pi - pj) == 1:  # Check for direct adjacency
                    page_continuity = 1.0
                elif pi == pj:  # Page compared to itself
                    page_continuity = 1.0  # Or based on self-similarity aspects, but usually text_sim is 1 for self.

                similarity_matrix[i, j] = (
                    0.4 * text_sim
                    + 0.2 * doc_type_sim
                    + 0.3 * header_sim
                    + 0.1 * page_continuity
                )
                if (
                    pi == pj
                ):  # Ensure self-similarity is high, mainly driven by text_sim being 1.
                    similarity_matrix[i, j] = max(
                        similarity_matrix[i, j], text_sim, 1.0
                    )  # Ensure self is 1.0

        return similarity_matrix, page_nums

    def compute_header_similarity(self, headers1, headers2):
        if not headers1 or not headers2:
            return 0.0

        max_sim = 0.0
        for h1_orig in headers1:
            for h2_orig in headers2:
                h1 = h1_orig.lower().strip()
                h2 = h2_orig.lower().strip()

                # Check for "(continued)" pattern for strong match
                h1_base = re.sub(r"\s*\(continued\)\s*$", "", h1).strip()
                h2_base = re.sub(r"\s*\(continued\)\s*$", "", h2).strip()

                # If base headers are identical and one or both are continuations, or if original headers are identical
                if h1_base == h2_base and (
                    h1.endswith("(continued)") or h2.endswith("(continued)") or h1 == h2
                ):
                    return 1.0  # Perfect match

                # Fallback to word-based similarity if not a direct continuation match
                words1 = set(h1.split())
                words2 = set(h2.split())

                if not words1 or not words2:
                    continue  # Skip if one header is empty after processing

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

            # current_matrix_idx is the index in similarity_matrix and page_nums for the last page added
            current_matrix_idx = i
            while current_matrix_idx < n_pages - 1:
                page_current_num = page_nums[current_matrix_idx]
                page_next_num = page_nums[current_matrix_idx + 1]

                doc_type_current = self.page_features[page_current_num]["doc_type"]
                doc_type_next = self.page_features[page_next_num]["doc_type"]

                force_break = False
                if (
                    doc_type_current != "unknown"
                    and doc_type_next != "unknown"
                    and doc_type_current != doc_type_next
                ):
                    force_break = True

                if (
                    not force_break
                    and similarity_matrix[current_matrix_idx, current_matrix_idx + 1]
                    >= threshold
                ):
                    current_cluster.append(page_next_num)
                    assigned_pages.add(page_next_num)
                    current_matrix_idx += (
                        1  # Move to the newly added page to check for the next one
                    )
                else:
                    break  # Break due to forced condition or low similarity

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

        all_providers = []
        all_facilities = []
        all_headers_from_all_pages = []
        doc_types_in_cluster = []

        if cluster:
            first_page_num = cluster[0]
            first_page_features = self.page_features.get(first_page_num)

            if first_page_features:
                first_page_dates = first_page_features["entities"]["dates"]
                if first_page_dates:
                    metadata["dos"] = max(
                        set(first_page_dates), key=first_page_dates.count
                    )

                if first_page_features["entities"]["headers"]:
                    metadata["header"] = first_page_features["entities"]["headers"][0]

        all_dates_overall = []
        for page_num in cluster:
            page_data = self.page_features.get(page_num)
            if not page_data:
                continue

            all_dates_overall.extend(page_data["entities"]["dates"])
            all_providers.extend(page_data["entities"]["providers"])
            all_facilities.extend(page_data["entities"]["facilities"])
            if page_data["entities"]["headers"]:
                all_headers_from_all_pages.extend(page_data["entities"]["headers"])
            doc_types_in_cluster.append(page_data["doc_type"])

        if not metadata["dos"] and all_dates_overall:
            metadata["dos"] = max(set(all_dates_overall), key=all_dates_overall.count)

        if not metadata["header"] and all_headers_from_all_pages:
            metadata["header"] = max(
                set(all_headers_from_all_pages), key=all_headers_from_all_pages.count
            )
        elif not metadata["header"]:
            metadata["header"] = "Unknown Header"

        if all_providers:
            metadata["provider"] = max(set(all_providers), key=all_providers.count)

        if all_facilities:
            metadata["facility"] = max(set(all_facilities), key=all_facilities.count)

        if doc_types_in_cluster:
            most_common_type = max(
                set(doc_types_in_cluster), key=doc_types_in_cluster.count
            )
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
        else:
            metadata["category"] = 0

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
