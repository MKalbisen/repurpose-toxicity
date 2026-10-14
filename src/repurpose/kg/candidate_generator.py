import numpy as np
import torch


def generate_candidates(model, metadata, entities, triples, disease_id, top_n=30, relation="indication"):
    if disease_id not in metadata["entity_to_id"]:
        raise ValueError("Hastalık eğitim çizgesinde yok")
    if relation != "indication" or relation not in metadata["relation_to_id"]:
        raise ValueError("Aday üretimi desteklenen indication ilişkisini gerektirir")
    drugs = entities[entities.type == "drug"].copy().sort_values("entity_id")
    drugs = drugs[drugs.entity_id.isin(metadata["entity_to_id"])]
    ids = [
        [
            metadata["entity_to_id"][d],
            metadata["relation_to_id"][relation],
            metadata["entity_to_id"][disease_id],
        ]
        for d in drugs.entity_id
    ]
    model.eval()
    with torch.no_grad():
        scores = (
            model.score_hrt(torch.tensor(ids, dtype=torch.long, device=model.device)).cpu().numpy().ravel()
        )
    drugs = drugs.rename(columns={"entity_id": "drug_id", "name": "drug_name"})
    drugs["kge_score"] = scores
    known = triples[(triples.relation == relation) & (triples["tail"] == disease_id)]["head"]
    reverse_known = triples[(triples.relation == relation) & (triples["head"] == disease_id)]["tail"]
    drugs["known_link"] = drugs.drug_id.isin(set(known) | set(reverse_known))
    drugs["disease_id"] = disease_id
    drugs = drugs.sort_values(["kge_score", "drug_id"], ascending=[False, True]).head(top_n)
    drugs["kge_rank"] = np.arange(1, len(drugs) + 1)
    return drugs[["drug_id", "drug_name", "disease_id", "kge_score", "kge_rank", "known_link"]].reset_index(
        drop=True
    )
