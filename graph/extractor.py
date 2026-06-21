import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import spacy
from spacy.pipeline import EntityRuler

from lexrag.ingestion.chunkers.base import Chunk
from lexrag.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int
    sentence: str

@dataclass
class Relation:
    source: str
    target: str
    type: str
    context_sentence: str
    
@dataclass
class GraphExtracts:
    entities: List[Entity]
    relations: List[Relation]
    source_chunk_id: str

class EntityRelationExtractor:
    """Extracts entities and relations for Graph RAG layer."""
    
    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.warning("en_core_web_lg not found, attempting to download...")
            from spacy.cli import download
            download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
            
        # Add legal patterns
        ruler_path = Path(__file__).parent / "legal_patterns.json"
        if ruler_path.exists():
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            with open(ruler_path, "r", encoding="utf-8") as f:
                patterns = json.load(f)
            ruler.add_patterns(patterns)
            
        self.target_labels = {"PERSON", "ORG", "DATE", "LAW", "MONEY", "GPE", "CONTRACT_PARTY"}
        
    def extract(self, chunks: List[Chunk]) -> List[GraphExtracts]:
        """Process chunks and extract entities and relationships."""
        logger.info("Extracting graph entities", chunk_count=len(chunks))
        
        all_extracts = []
        for chunk in chunks:
            doc = self.nlp(chunk.text)
            
            entities = []
            for ent in doc.ents:
                if ent.label_ in self.target_labels:
                    entities.append(Entity(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        sentence=ent.sent.text
                    ))
                    
            relations = []
            # Extract relations using dependency parsing within sentences
            for sent in doc.sents:
                sent_ents = [e for e in entities if e.sentence == sent.text]
                if len(sent_ents) >= 2:
                    # Simple heuristic: pair every entity in the sentence
                    # In a full system, you trace the dependency path between the two entity tokens
                    for i, e1 in enumerate(sent_ents):
                        for e2 in sent_ents[i+1:]:
                            # Simplified relation extraction based on root verb between entities
                            root_verb = "RELATED_TO"
                            for token in sent:
                                if token.pos_ == "VERB" and e1.start_char <= token.idx <= e2.end_char:
                                    root_verb = token.lemma_.upper()
                                    break
                            relations.append(Relation(
                                source=e1.text,
                                target=e2.text,
                                type=root_verb,
                                context_sentence=sent.text
                            ))
                            
            all_extracts.append(GraphExtracts(
                entities=entities,
                relations=relations,
                source_chunk_id=chunk.chunk_id
            ))
            
        return all_extracts
