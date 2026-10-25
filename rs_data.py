from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdDepictor

MoleculeRecord = Dict[str, Any]

@dataclass(frozen=True)
class StereoGenerator:
    molecules: Mapping[str, MoleculeRecord] | None = None

    def __post_init__(self) -> None:
        if self.molecules is None:
            mols = copy.deepcopy(_MOLECULES)
            
            for name, record in mols.items():
                mol = Chem.MolFromSmiles(record["smiles"])
                rdDepictor.Compute2DCoords(mol)
                rdCIPLabeler.AssignCIPLabels(mol)
                
                # Extract chiral centers [(atom_idx, 'R' or 'S'), ...]
                centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
                record["chiral_centers"] = {str(idx): code for idx, code in centers}
                
            object.__setattr__(self, "molecules", mols)

    def get_random_molecule(self) -> MoleculeRecord:
        assert self.molecules is not None
        name = random.choice(list(self.molecules))
        record = copy.deepcopy(self.molecules[name])
        record["name"] = name
        return record

    def get_molecule(self, name: str) -> MoleculeRecord:
        assert self.molecules is not None
        record = copy.deepcopy(self.molecules[name])
        record["name"] = name
        return record

    def list_molecules(self) -> List[str]:
        assert self.molecules is not None
        return sorted(self.molecules)

_MOLECULES: Dict[str, MoleculeRecord] = {
    "2-chlorobutane (Isomer A)": {"smiles": "CC[C@@H](C)Cl", "description": "Fun Fact: Highly flammable and has a sharp, ether-like odor."},
    "2-chlorobutane (Isomer B)": {"smiles": "CC[C@H](C)Cl", "description": "Fun Fact: Used extensively as an industrial solvent and chemical intermediate."},
    "Tartaric acid (Isomer A)": {"smiles": "O=C(O)[C@@H](O)[C@@H](O)C(=O)O", "description": "Fun Fact: This compound is responsible for the tartness of wine and is found naturally in grapes."},
    "Tartaric acid (Isomer B)": {"smiles": "O=C(O)[C@H](O)[C@H](O)C(=O)O", "description": "Fun Fact: Louis Pasteur famously separated the crystals of this molecule with tweezers in 1848!"},
    "Tartaric acid (Isomer C)": {"smiles": "O=C(O)[C@@H](O)[C@H](O)C(=O)O", "description": "Fun Fact: Used heavily in the food industry as an additive to give sour candies their bite."},
    "Alanine": {"smiles": "N[C@@H](C)C(=O)O", "description": "Fun Fact: This amino acid is highly concentrated in silk fibroin, the protein that makes up spider webs."},
    "Glyceraldehyde": {"smiles": "O=C[C@@H](O)CO", "description": "Fun Fact: Found in the human body as an intermediate during the breakdown of sugars for cellular energy."},
    "Lactic acid": {"smiles": "C[C@@H](O)C(=O)O", "description": "Fun Fact: Gives sourdough bread and lambic beer their distinctive tangy flavor."},
    "Carvone (Spearmint)": {"smiles": "CC(=C)[C@@H]1CCC(=O)C=C1", "description": "Fun Fact: Smells strongly of spearmint and is heavily used to flavor chewing gum and toothpaste."},
    "Carvone (Caraway)": {"smiles": "CC(=C)[C@H]1CCC(=O)C=C1", "description": "Fun Fact: Smells strongly of caraway seeds and is used to flavor rye bread and certain cheeses."},
    "Cyclohexane-1,2-diol": {"smiles": "O[C@@H]1CCCC[C@H]1O", "description": "Fun Fact: Used as a chemical building block for producing certain types of plastics and nylon."},
    "3-methylpentan-2-ol": {"smiles": "CC[C@H](C)[C@@H](C)O", "description": "Fun Fact: A branched alcohol often utilized in the fragrance and flavoring industry."},
    "Ibuprofen": {"smiles": "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O", "description": "Fun Fact: This molecule is the active ingredient in popular over-the-counter NSAID pain relievers."},
    "2-bromopropanoic acid": {"smiles": "C[C@@H](Br)C(=O)O", "description": "Fun Fact: Used to manufacture various agricultural chemicals, including herbicides."},
    "3-chlorobutan-2-ol (Isomer A)": {"smiles": "C[C@@H](O)[C@@H](Cl)C", "description": "Fun Fact: A clear, colorless liquid with a strong, pungent odor."},
    "3-chlorobutan-2-ol (Isomer B)": {"smiles": "C[C@H](O)[C@@H](Cl)C", "description": "Fun Fact: Typically handled in fume hoods due to its strong vapor."},
    "Epichlorohydrin": {"smiles": "C1[C@H](O1)CCl", "description": "Fun Fact: A highly reactive building block used to manufacture epoxy resins and superglue."},
    "Malic acid": {"smiles": "O=C(O)C[C@@H](O)C(=O)O", "description": "Fun Fact: The compound responsible for the extreme sour taste of green apples and Warheads candy!"},
    "2,3-dichlorobutane (Isomer A)": {"smiles": "C[C@@H](Cl)[C@@H](Cl)C", "description": "Fun Fact: Used as an industrial solvent to remove heavy greases and oils."},
    "2,3-dichlorobutane (Isomer B)": {"smiles": "C[C@@H](Cl)[C@H](Cl)C", "description": "Fun Fact: Commonly used in the manufacturing of synthetic rubber."},
    "Nicotine": {"smiles": "CN1CCC[C@H]1C2=CN=CC=C2", "description": "Fun Fact: Acts as a natural botanical insecticide for the tobacco plant."},
    "Limonene (Lemon)": {"smiles": "CC(=C)[C@@H]1CCC(=CC1)C", "description": "Fun Fact: The major component in the oil of citrus fruit peels."},
    "Limonene (Pine)": {"smiles": "CC(=C)[C@H]1CCC(=CC1)C", "description": "Fun Fact: This molecule smells distinctly like pine needles and turpentine rather than citrus."},
    "Menthol": {"smiles": "C[C@@H]1CC[C@H]([C@@H](C)C)[C@H](O)C1", "description": "Fun Fact: Triggers the cold-sensitive TRPM8 receptors in the skin, tricking your brain into feeling a cooling sensation."},
    "Amphetamine": {"smiles": "C[C@@H](N)CC1=CC=CC=C1", "description": "Fun Fact: First synthesized in 1887 by a Romanian chemist working in Germany."},
    "2-butanol (Isomer A)": {"smiles": "CC[C@@H](C)O", "description": "Fun Fact: A common industrial solvent used to thin paints and varnishes."},
    "2-butanol (Isomer B)": {"smiles": "CC[C@H](C)O", "description": "Fun Fact: Found in trace amounts in some naturally fermented fruit juices."},
    "2-methylcyclohexanol (Isomer A)": {"smiles": "C[C@@H]1CCCC[C@H]1O", "description": "Fun Fact: Used as a specialty solvent for certain plastics and resins."},
    "2-methylcyclohexanol (Isomer B)": {"smiles": "C[C@H]1CCCC[C@@H]1O", "description": "Fun Fact: Often added to industrial degreasers and spot removers."},
    "Cysteine (Isomer A)": {"smiles": "N[C@@H](CS)C(=O)O", "description": "Fun Fact: Forms strong disulfide bonds that give curly hair its shape!"},
    "Cysteine (Isomer B)": {"smiles": "N[C@H](CS)C(=O)O", "description": "Fun Fact: Used by the food industry as a dough conditioner in commercial bread baking."},
    "Butane-2,3-diol (Isomer A)": {"smiles": "C[C@@H](O)[C@@H](O)C", "description": "Fun Fact: Used industrially in the production of synthetic rubber and antifreeze."},
    "Butane-2,3-diol (Isomer B)": {"smiles": "C[C@H](O)[C@H](O)C", "description": "Fun Fact: Produced naturally by certain strains of Klebsiella bacteria during fermentation."},
    "Butane-2,3-diol (Isomer C)": {"smiles": "C[C@@H](O)[C@H](O)C", "description": "Fun Fact: Often used as an additive in the manufacturing of printing inks and perfumes."},
    "Thalidomide": {"smiles": "O=C1NC(=O)C(CC1)N2C(=O)c3ccccc3C2=O", "description": "Fun Fact: Originally marketed in the 1950s as a sedative and anti-nausea medication."},
    "Proline": {"smiles": "O=C(O)[C@@H]1CCCN1", "description": "Fun Fact: An essential component of collagen, the main structural protein in human skin and connective tissue."},
    "1,2-dibromocyclopentane (Isomer A)": {"smiles": "Br[C@@H]1CCC[C@@H]1Br", "description": "Fun Fact: A heavy, dense liquid that sinks quickly when added to water."},
    "1,2-dibromocyclopentane (Isomer B)": {"smiles": "Br[C@@H]1CCC[C@H]1Br", "description": "Fun Fact: Used in chemical research as a highly dense organic solvent."},
    "sec-Butylamine": {"smiles": "CC[C@@H](C)N", "description": "Fun Fact: A clear organic liquid with a strong, fishy ammonia-like odor."},
    "Camphor": {"smiles": "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2", "description": "Fun Fact: Found in Vicks VapoRub and historically used to make early plastics like celluloid!"},
    "3-chloro-1-butene (Isomer A)": {"smiles": "C=C[C@@H](Cl)C", "description": "Fun Fact: A highly volatile liquid that boils almost exactly at room temperature."},
    "3-chloro-1-butene (Isomer B)": {"smiles": "C=C[C@H](Cl)C", "description": "Fun Fact: Must be stored in tightly sealed containers to prevent rapid evaporation."},
    "1-phenylethanol (Isomer A)": {"smiles": "C[C@@H](O)c1ccccc1", "description": "Fun Fact: Found in nature as a minor floral component of tea and essential oils."},
    "1-phenylethanol (Isomer B)": {"smiles": "C[C@H](O)c1ccccc1", "description": "Fun Fact: Has a mild, pleasant floral scent similar to hyacinths."},
    "Epoxybutane (Isomer A)": {"smiles": "C[C@@H]1O[C@@H]1C", "description": "Fun Fact: Highly flammable and must be kept strictly away from open flames."},
    "Epoxybutane (Isomer B)": {"smiles": "C[C@H]1O[C@H]1C", "description": "Fun Fact: Emits a sweet, ether-like odor."},
    "Epoxybutane (Isomer C)": {"smiles": "C[C@@H]1O[C@H]1C", "description": "Fun Fact: Often used as an industrial stabilizer for chlorinated solvents."},
    "Aspartic acid": {"smiles": "N[C@@H](CC(=O)O)C(=O)O", "description": "Fun Fact: One of the two amino acids that make up the artificial sweetener Aspartame."},
    "2-phenylpropanoic acid": {"smiles": "C[C@@H](c1ccccc1)C(=O)O", "description": "Fun Fact: Found in the essential oils of some rare tropical plants."},
    "1,3-dimethylcyclohexane": {"smiles": "C[C@@H]1CCC[C@@H](C)C1", "description": "Fun Fact: Naturally found in crude oil and refined petroleum products."}
}