import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

from neo4j import AsyncGraphDatabase
from rapidfuzz import fuzz

from lexrag.config import settings
from lexrag.logger import get_logger
from .extractor import Entity, Relation

logger = get_logger(__name__)

@dataclass
class GraphQueryResult:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    contexts: List[str]

class Neo4jGraphStore:
    """Graph database store for extracted entities and relations."""
    
    def __init__(self) -> None:
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.known_entities: Dict[str, str] = {} # cache for disambiguation
        
    async def create_constraints(self) -> None:
        """Create uniqueness constraints in Neo4j."""
        async with self.driver.session() as session:
            # Need to use try/except as constraint might already exist
            try:
                await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                logger.info("Neo4j constraints created")
            except Exception as e:
                logger.warning(f"Error creating constraints: {e}")

    async def _disambiguate(self, entity_name: str) -> str:
        """Simple entity disambiguation using RapidFuzz."""
        normalized = entity_name.lower()
        for suffix in [" inc.", " corp.", " ltd.", " llc.", " corporation"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                
        # Check against known entities
        for known_name, known_norm in self.known_entities.items():
            if fuzz.ratio(normalized, known_norm) > 92:
                return known_name
                
        self.known_entities[entity_name] = normalized
        return entity_name

    async def upsert_entity(self, entity: Entity, doc_id: str) -> None:
        """Upsert a single entity into the graph."""
        canonical_name = await self._disambiguate(entity.text)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.first_seen = $now, e.document_ids = [$doc_id]
        ON MATCH SET e.document_ids = CASE WHEN $doc_id IN e.document_ids THEN e.document_ids ELSE e.document_ids + [$doc_id] END
        """
        async with self.driver.session() as session:
            await session.run(query, name=canonical_name, type=entity.label, now=now, doc_id=doc_id)

    async def upsert_relation(self, relation: Relation) -> None:
        """Upsert a relationship between two entities."""
        src_name = await self._disambiguate(relation.source)
        tgt_name = await self._disambiguate(relation.target)
        
        query = """
        MATCH (a:Entity {name: $from_name})
        MATCH (b:Entity {name: $to_name})
        MERGE (a)-[r:RELATED_TO {type: $rel_type}]->(b)
        ON CREATE SET r.contexts = [$context]
        ON MATCH SET r.contexts = CASE WHEN $context IN r.contexts THEN r.contexts ELSE r.contexts + [$context] END
        """
        async with self.driver.session() as session:
            await session.run(
                query, 
                from_name=src_name, 
                to_name=tgt_name, 
                rel_type=relation.type, 
                context=relation.context_sentence
            )

    async def get_entity_neighbors(self, entity_name: str, hops: int = 2) -> GraphQueryResult:
        """Retrieve graph neighbors up to N hops away."""
        canonical_name = await self._disambiguate(entity_name)
        
        query = f"""
        MATCH path = (e:Entity {{name: $name}})-[*1..{hops}]-(neighbor:Entity)
        RETURN path, neighbor
        LIMIT 50
        """
        nodes = []
        edges = []
        contexts = []
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, name=canonical_name)
                records = await result.data()
                
                for record in records:
                    nodes.append(record.get("neighbor", {}))
                    # In a real app we'd parse the path relationships correctly here
                    # For now we'll just extract contexts if available
                    path = record.get("path", [])
                    # We expect path to have relationships that contain 'contexts'
        except Exception as e:
            logger.error("Failed to fetch entity neighbors", entity=entity_name, error=str(e))
            
        return GraphQueryResult(nodes=nodes, edges=edges, contexts=list(set(contexts)))

    async def close(self) -> None:
        """Close the database driver."""
        await self.driver.close()
