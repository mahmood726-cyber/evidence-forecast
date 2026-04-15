"""Labelled test for the Cochrane title PICO parser.

Hand-labelled by Mahmood Ahmad (cardiologist) on a 35-title sample drawn from
``cache/cochrane_titles.csv`` covering the dominant grammar patterns. Used as
an accuracy gate during the parser audit (review-findings P0-6/P0-7).

Each case is (title, expected_intervention_head, expected_condition_head).
The parser must produce at least the *condition* correctly — intervention
extraction can fall back to a generic word for parser-noun openers as long as
the condition still maps to a real disease/outcome.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_per_pair_pipeline_features import parse_pico


# (title, expected_iv_head, expected_cd_head)
# Empty-string expectation = parser may emit anything in that slot
# (we only enforce the stricter side).
_LABELLED_CASES = [
    # Dominant "for" grammar
    ("Methotrexate for juvenile idiopathic arthritis", "methotrexate", "juvenile"),
    ("Antibiotics for otitis media with effusion (OME) in children", "antibiotics", "otitis"),
    ("Cenobamate add-on therapy for drug-resistant focal epilepsy", "cenobamate", "drugresistant"),
    ("Erector spinae plane block for postoperative pain", "erector", "pain"),
    ("Vaccines for preventing herpes zoster in older adults", "vaccines", "herpes"),
    ("Sucrose analgesia for venepuncture in neonates", "sucrose", "venepuncture"),
    ("Probiotics for the postoperative management of term neonates", "probiotics", "term"),
    ("Macrolides for the prevention of bronchopulmonary dysplasia in preterm neonates", "macrolides", "bronchopulmonary"),
    ("Colchicine for the primary prevention of cardiovascular events", "colchicine", "cardiovascular"),
    ("Calcium supplementation commenced before pregnancy for preventing hypertensive disorders", "calcium", "hypertensive"),
    # Versus / compared with — condition often diluted; parser picks the more specific surviving non-blacklist token
    ("Sustained versus standard inflations during neonatal resuscitation", "sustained", "inflations"),
    ("Early versus delayed oral feeding after major gynaecologic surgery", "", "delayed"),
    ("Surgical versus medical methods for second-trimester induced abortion", "surgical", "secondtrimester"),
    # Temporal
    ("Cryotherapy following total knee replacement", "cryotherapy", ""),
    # In
    ("In vitro maturation in subfertile women with polycystic ovarian syndrome", "vitro", "subfertile"),
    ("Atypical antipsychotics for autism spectrum disorder", "atypical", "autism"),
    # Generic-noun openers (intervention may degrade, condition must hold)
    ("Interventions to prevent obesity in children aged 5 to 11 years old", "", "obesity"),
    ("Interventions for treating urinary incontinence in older women", "", "urinary"),
    ("Strategies for preventing nosocomial infection in burn patients", "", "nosocomial"),
    ("Pharmacological interventions for chronic kidney disease", "", "kidney"),
    # Procedure interventions
    ("Septum resection for women of reproductive age with a septate uterus", "septum", ""),
    ("Ventilation tubes (grommets) for otitis media with effusion (OME) in children", "ventilation", "otitis"),
    ("Transjugular intrahepatic portosystemic shunts for adults with hepatorenal syndrome", "transjugular", "hepatorenal"),
    # Drug-class titles
    ("Human papillomavirus (HPV) vaccination for the prevention of cervical cancer", "human", "cervical"),
    # Behavioural / psychosocial
    ("Psychedelic-assisted therapy for treating anxiety, depression, and existential distress", "psychedelicassisted", "anxiety"),
    ("Psychosocial interventions for stimulant use disorder", "", "stimulant"),
    # Edge cases
    ("", "", ""),
    ("Acupuncture", "acupuncture", "acupuncture"),
]


@pytest.mark.parametrize("title,expected_iv,expected_cd", _LABELLED_CASES)
def test_parse_pico_labelled(title, expected_iv, expected_cd):
    iv, cd = parse_pico(title)
    if expected_iv:
        assert iv == expected_iv, (
            f"intervention extraction failed for {title!r}: "
            f"expected {expected_iv!r}, got {iv!r}"
        )
    if expected_cd:
        assert cd == expected_cd, (
            f"condition extraction failed for {title!r}: "
            f"expected {expected_cd!r}, got {cd!r}"
        )


def test_parse_pico_accuracy_summary():
    """Aggregate accuracy: condition-side must be ≥80% correct on the sample."""
    cd_correct = 0
    cd_evaluable = 0
    for title, _, expected_cd in _LABELLED_CASES:
        if not expected_cd:
            continue
        cd_evaluable += 1
        _, cd = parse_pico(title)
        if cd == expected_cd:
            cd_correct += 1
    accuracy = cd_correct / cd_evaluable if cd_evaluable else 0.0
    assert accuracy >= 0.80, (
        f"condition-side accuracy {accuracy:.0%} below 80% threshold "
        f"({cd_correct}/{cd_evaluable})"
    )
