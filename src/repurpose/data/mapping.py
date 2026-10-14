"""PubChem structure lookup with DrugBank identifier verification and local overrides."""

import re
import time
from urllib.parse import quote

import pandas as pd
import requests

from repurpose.data.chemistry import canonicalize
from repurpose.utils import ROOT, paths, write_json

COLS = [
    "drug_id",
    "drug_name",
    "drugbank_id",
    "pubchem_cid",
    "cas_ids",
    "chembl_ids",
    "smiles",
    "canonical_smiles",
    "source",
    "verified",
    "status",
    "salt_fragments",
    "stereochemistry",
    "reason",
]
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def resolve_pubchem(drug_id, drug_name, drugbank_id):
    row = dict.fromkeys(COLS)
    row.update(
        drug_id=drug_id,
        drug_name=drug_name,
        drugbank_id=drugbank_id,
        verified=False,
        status="unmatched",
        source="PubChem PUG REST",
    )
    try:
        # Resolve exact source identifier first, never silently trust a name alone.
        r = requests.get(
            f"{PUG}/compound/name/{quote(drugbank_id, safe='')}/property/SMILES/JSON", timeout=20
        )
        if r.status_code == 404:
            r = requests.get(
                f"{PUG}/compound/name/{quote(drug_name, safe='')}/property/SMILES/JSON", timeout=20
            )
        r.raise_for_status()
        props = r.json()["PropertyTable"]["Properties"]
        if len(props) != 1:
            row["reason"] = "multiple PubChem structures"
            return row
        prop = props[0]
        cid = prop["CID"]
        synonyms = requests.get(f"{PUG}/compound/cid/{cid}/synonyms/JSON", timeout=20)
        synonyms.raise_for_status()
        names = synonyms.json()["InformationList"]["Information"][0]["Synonym"]
        verified = drugbank_id.upper() in {x.upper() for x in names}
        smi = prop.get("SMILES") or prop.get("IsomericSMILES") or prop.get("ConnectivitySMILES")
        canonical = canonicalize(smi)
        row.update(
            pubchem_cid=cid,
            cas_ids=";".join(sorted({x for x in names if re.fullmatch(r"\d{2,7}-\d{2}-\d", x)})),
            chembl_ids=";".join(sorted({x for x in names if re.fullmatch(r"CHEMBL\d+", x)})),
            smiles=smi,
            canonical_smiles=canonical,
            verified=verified and canonical is not None,
            status="matched" if verified and canonical else "unverified",
            salt_fragments=smi.count(".") + 1 if smi else None,
            stereochemistry="retained",
            reason="" if verified else "DrugBank ID not in PubChem synonyms",
        )
    except (requests.RequestException, KeyError, ValueError) as e:
        row["reason"] = str(e)[:240]
    return row


def map_drugs(config):
    base, _, reports = paths(config)
    entities = pd.read_csv(base / "entities.csv", dtype=str)
    drugs = entities[entities.type == "drug"].sort_values("entity_id")
    destination = base / "drug_chemical_mapping.csv"
    existing = pd.read_csv(destination).to_dict("records") if destination.exists() else []
    # User-curated mappings must include source and verified fields, never just names.
    overrides = ROOT / "data/mappings/verified_drugs.csv"
    if overrides.exists():
        custom = pd.read_csv(overrides)
        if not {"drug_id", "smiles", "source", "verified"} <= set(custom):
            raise ValueError("Manuel eşleme: drug_id, smiles, source, verified gerekli")
        if custom.drug_id.duplicated().any():
            custom[custom.drug_id.duplicated(keep=False)].to_csv(
                base / "ambiguous_drug_mappings.csv", index=False
            )
            raise ValueError(
                "Bir drug_id için çoklu manuel SMILES; ambiguous_drug_mappings.csv içinden tek kimlik seçin"
            )
        if custom.source.isna().any() or custom.source.astype(str).str.strip().eq("").any():
            raise ValueError("Kimyasal eşleme kaynak bilgisi boş olamaz")
        for item in custom.to_dict("records"):
            item["canonical_smiles"] = canonicalize(item["smiles"])
            item["status"] = (
                "matched"
                if str(item["verified"]).lower() == "true" and item["canonical_smiles"]
                else "unverified"
            )
            existing = [r for r in existing if r["drug_id"] != item["drug_id"]]
            existing.append(item)
    seen = {r["drug_id"] for r in existing if r.get("status") == "matched"}
    limit = config["mapping"]["limit"]
    queried = 0
    for d in drugs.itertuples(index=False):
        if d.entity_id in seen:
            continue
        if limit is not None and queried >= limit:
            break
        existing = [r for r in existing if r["drug_id"] != d.entity_id]
        existing.append(resolve_pubchem(d.entity_id, d.name, d.source_id))
        queried += 1
        time.sleep(0.25)  # Below PubChem's request rate limit.
        pd.DataFrame(existing).reindex(columns=COLS).to_csv(destination, index=False)
    mapped = pd.DataFrame(existing).reindex(columns=COLS)
    all_drugs = drugs[["entity_id", "name"]].rename(columns={"entity_id": "drug_id", "name": "drug_name"})
    unmatched = all_drugs.merge(mapped.drop(columns="drug_name"), on="drug_id", how="left")
    unmatched = unmatched[unmatched.status != "matched"]
    unmatched.to_csv(base / "unmatched_drugs.csv", index=False)
    mapped.to_csv(destination, index=False)
    write_json(
        reports / "mapping_report.json",
        {
            "total_drugs": len(drugs),
            "queried_this_run": queried,
            "verified_structures": int((mapped.status == "matched").sum()),
            "unmatched": len(unmatched),
            "policy": "DrugBank ID must appear in PubChem CID synonyms or explicit curated verification.",
        },
    )
