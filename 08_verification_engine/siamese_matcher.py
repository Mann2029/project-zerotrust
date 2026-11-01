import os
import glob
import numpy as np
import time
from typing import Dict, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, Exception):
    FAISS_AVAILABLE = False

class SiameseMatcher:
    def __init__(self, embeddings_dir: str, embedding_dim: int = 512):
        """
        Initializes the Advanced Siamese Neural Network (SNN) Matcher.
        Uses FAISS (Facebook AI Similarity Search) for instantaneous 1-to-N matching,
        perfectly synced with the Custom Multi-Task CNN 512-D outputs.
        """
        self.embeddings_dir = embeddings_dir
        self.embedding_dim = embedding_dim
        
        # FAISS Index (Inner Product == Cosine Similarity for L2 Normalized vectors)
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = None
            
        self.registry_matrix = None
        
        # Mapping dictionaries (FAISS only understands integer IDs)
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.current_id_counter = 0
        
        self.reload_registry()

    @property
    def registry_keys(self) -> list:
        """Returns the list of all registered identity keys online in RAM."""
        return list(self.name_to_id.keys())


    def add_embedding_to_ram(self, identity: str, vector: np.ndarray):
        """
        Incrementally injects a single new registration directly into RAM.
        Eliminates the need to reload the hard drive when a new user registers.
        """
        vector = np.array(vector, dtype=np.float32).flatten()
        
        # 1. Strict Dimension Verification (Protects against CNN crashes)
        if vector.shape[0] != self.embedding_dim:
            print(f"[SNN ERROR] Rejected {identity}: Expected {self.embedding_dim}-D vector, got {vector.shape[0]}-D.")
            return

        # 2. L2 Normalization Enforcement
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            print(f"[SNN ERROR] Rejected {identity}: Zero vector.")
            return

        reshaped_vector = np.expand_dims(vector, axis=0)

        # 3. Add to Mapping
        if identity not in self.name_to_id:
            internal_id = self.current_id_counter
            self.name_to_id[identity] = internal_id
            self.id_to_name[internal_id] = identity
            self.current_id_counter += 1
            if self.registry_matrix is None:
                self.registry_matrix = reshaped_vector
            else:
                self.registry_matrix = np.vstack([self.registry_matrix, reshaped_vector])
        else:
            internal_id = self.name_to_id[identity]
            self.registry_matrix[internal_id] = vector

        # 4. Inject into Engine
        if FAISS_AVAILABLE:
            # IndexFlatIP has no update operation; rebuild it so replacement
            # registrations cannot leave stale vectors behind.
            self.index.reset()
            self.index.add(self.registry_matrix)

    def reload_registry(self):
        """
        Cold-boot sequence. Only runs once when the server starts.
        Scans all .npy files and bulk-loads them into FAISS.
        """
        print(f"[SNN] Booting Advanced Vector Engine from {self.embeddings_dir}...")
        start_time = time.perf_counter()
        
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir, exist_ok=True)
            
        npy_files = glob.glob(os.path.join(self.embeddings_dir, "*.npy"))
        
        # Reset Indexes
        if FAISS_AVAILABLE:
            self.index.reset()
        self.registry_matrix = None
        self.id_to_name.clear()
        self.name_to_id.clear()
        self.current_id_counter = 0

        for file_path in npy_files:
            try:
                filename = os.path.basename(file_path)
                if filename.startswith("vector_"):
                    identity = filename.replace("vector_", "").replace(".npy", "")
                    vector = np.load(file_path)
                    self.add_embedding_to_ram(identity, vector)
                elif filename.startswith("faculty_"):
                    identity = filename.replace(".npy", "")
                    vector = np.load(file_path)
                    self.add_embedding_to_ram(identity, vector)
                
            except Exception as e:
                print(f"[SNN ERROR] Failed to load {file_path}: {e}")
                
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        engine_type = "FAISS C++" if FAISS_AVAILABLE else "Numpy Fallback"
        print(f"[SUCCESS] SNN {engine_type} Matrix Locked: {len(self.id_to_name)} identities online in {elapsed_ms:.2f}ms.")

    def verify_identity(
        self, 
        live_vector: np.ndarray, 
        threshold: float = 0.65,
        enrolled_cohort: list = None,
        allow_cross_section: bool = True
    ) -> Dict[str, Any]:
        """
        Lightning-fast 1-to-N similarity search with intelligent cohort-aware disambiguation.
        Matches the 512-D live stream vector against registered templates, protecting enrolled
        cohort members from cross-talk misidentification while still supporting cross-section routing.
        """
        if live_vector is None:
            return {"match_found": False, "reason": "Input embedding vector is None."}

        if len(self.id_to_name) == 0:
            return {"match_found": False, "reason": "Database is empty."}
            
        try:
            live_vector = np.array(live_vector, dtype=np.float32).flatten()
            
            # Strict Dimension Check
            if live_vector.shape[0] != self.embedding_dim:
                return {"match_found": False, "reason": f"Dimension mismatch. Expected {self.embedding_dim}-D."}

            norm = np.linalg.norm(live_vector)
            if norm > 0:
                live_vector = live_vector / norm
            else:
                return {"match_found": False, "reason": "Invalid zero-vector received."}

            similarities = np.dot(self.registry_matrix, live_vector)
            cohort_set = set(enrolled_cohort) if enrolled_cohort else set()
            cohort_candidates = []
            other_candidates = []

            for idx, sim in enumerate(similarities):
                ident = self.id_to_name.get(idx, "UNKNOWN")
                if ident.startswith("faculty_"):
                    continue  # Exclude faculty templates from student attendance checks
                if cohort_set and ident in cohort_set:
                    cohort_candidates.append((ident, float(sim)))
                else:
                    other_candidates.append((ident, float(sim)))

            cohort_candidates.sort(key=lambda x: x[1], reverse=True)
            other_candidates.sort(key=lambda x: x[1], reverse=True)

            best_cohort_id, best_cohort_sim = cohort_candidates[0] if cohort_candidates else ("UNKNOWN", -1.0)
            best_other_id, best_other_sim = other_candidates[0] if other_candidates else ("UNKNOWN", -1.0)

            # Smart cohort disambiguation
            if cohort_set and best_cohort_id != "UNKNOWN":
                if best_other_sim > best_cohort_sim:
                    # Outside candidate only takes precedence if decisively superior with high confidence
                    if allow_cross_section and best_other_sim >= 0.78 and (best_other_sim - best_cohort_sim) >= 0.12:
                        best_identity = best_other_id
                        highest_similarity = best_other_sim
                    elif best_cohort_sim >= 0.55:
                        # Enrolled cohort candidate prioritized over ambiguous cross-section noise
                        best_identity = best_cohort_id
                        highest_similarity = best_cohort_sim
                    else:
                        best_identity = best_other_id
                        highest_similarity = best_other_sim
                else:
                    best_identity = best_cohort_id
                    highest_similarity = best_cohort_sim
            else:
                # No cohort provided: fallback to global best match
                best_match_idx = int(np.argmax(similarities))
                best_identity = self.id_to_name.get(best_match_idx, "UNKNOWN")
                highest_similarity = float(similarities[best_match_idx])
            
            # ---------------------------------------------------------
            # Result Validation
            # ---------------------------------------------------------
            if highest_similarity >= threshold:
                return {
                    "match_found": True,
                    "identity": best_identity,
                    "registration_no": best_identity,
                    "candidate_identity": best_identity,
                    "similarity": round(highest_similarity, 4),           # Decimal (for main.py)
                    "confidence": round(highest_similarity * 100.0, 2)    # Percentage (for WebSockets)
                }
            else:
                return {
                    "match_found": False,
                    "identity": "UNKNOWN",
                    "registration_no": "UNKNOWN",
                    "candidate_identity": best_identity,
                    "similarity": round(highest_similarity, 4),
                    "confidence": round(highest_similarity * 100.0, 2),
                    "reason": f"Match ({highest_similarity:.4f}) failed threshold ({threshold})"
                }
        except Exception as e:
            print(f"[SNN ERROR] Verification crashed: {e}")
            return {"match_found": False, "reason": str(e)}

    def find_best_match(self, live_vector: np.ndarray, threshold: float = 0.65, enrolled_cohort: list = None):
        """Universal compatibility alias for verify_identity."""
        res = self.verify_identity(live_vector, threshold=threshold, enrolled_cohort=enrolled_cohort)
        return res.get("match_found", False), res.get("similarity", 0.0), res.get("identity", "UNKNOWN")

    def verify_faculty(self, live_vector: np.ndarray, faculty_code: str = "EMP-AIML-01", threshold: float = 0.58) -> Dict[str, Any]:
        """
        Dedicated 1-to-1 verification for faculty camera unlock.
        Compares live 512-D vector against the registered faculty template.
        """
        if live_vector is None:
            return {"match_found": False, "similarity": 0.0, "reason": "Input vector is None"}

        target_key = f"faculty_{faculty_code}" if not faculty_code.startswith("faculty_") else faculty_code
        faculty_vec = None
        if target_key in self.name_to_id and self.registry_matrix is not None:
            idx = self.name_to_id[target_key]
            faculty_vec = self.registry_matrix[idx]
        else:
            file_path = os.path.join(self.embeddings_dir, f"{target_key}.npy")
            if os.path.exists(file_path):
                faculty_vec = np.load(file_path).flatten()
                norm = np.linalg.norm(faculty_vec)
                if norm > 0:
                    faculty_vec = faculty_vec / norm

        if faculty_vec is None:
            return {"match_found": False, "similarity": 0.0, "reason": f"Faculty template {target_key} not found"}

        live_vec = np.array(live_vector, dtype=np.float32).flatten()
        norm_live = np.linalg.norm(live_vec)
        if norm_live > 0:
            live_vec = live_vec / norm_live
        else:
            return {"match_found": False, "similarity": 0.0, "reason": "Zero live vector"}

        sim = float(np.dot(faculty_vec, live_vec))
        return {
            "match_found": bool(sim >= threshold),
            "similarity": round(sim, 4),
            "confidence": round(sim * 100.0, 1),
            "faculty_id": faculty_code
        }
