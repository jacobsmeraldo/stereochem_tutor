from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Dict, Iterable, Mapping

import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdCIPLabeler, rdDepictor

from rs_data import StereoGenerator
from rs_tutor import get_stereo_tutor_response

st.set_page_config(page_title="R/S Stereocenter Drill", layout="wide")

def smiles_to_stereo_mol(smiles: str, chiral_indices: Iterable[str]) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    rdDepictor.Compute2DCoords(mol)
    rdCIPLabeler.AssignCIPLabels(mol)

    chiral_set = {int(idx) for idx in chiral_indices}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in chiral_set:
            atom.SetProp("atomNote", f"C{atom.GetIdx()}")
    return mol

def molecule_image(mol: Chem.Mol) -> BytesIO:
    image = Draw.MolToImage(mol, size=(520, 420), kekulize=True)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def get_3d_mol_block(smiles: str) -> str:
    """Generates a standard 3D geometry block for the interactive viewer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True) 
        
    return Chem.MolToMolBlock(mol)

def clear_chat() -> None:
    st.session_state.chat_history = []
    for key in list(st.session_state):
        if key.startswith("stereo_"):
            del st.session_state[key]

def new_problem(generator: StereoGenerator) -> None:
    st.session_state.problem_data = generator.get_random_molecule()
    clear_chat()

def initialize_state(generator: StereoGenerator) -> None:
    if "problem_data" not in st.session_state:
        st.session_state.problem_data = generator.get_random_molecule()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def hydrate_secret_api_key() -> None:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None

    if secret_key and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = secret_key

def render_chat() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main() -> None:
    hydrate_secret_api_key()
    generator = StereoGenerator()
    initialize_state(generator)

    problem = st.session_state.problem_data
    chiral_centers = problem.get("chiral_centers", {})
    mol = smiles_to_stereo_mol(problem["smiles"], chiral_centers.keys())

    st.title("R/S Stereocenter Drill")
    st.caption("Determine the Cahn-Ingold-Prelog (R or S) absolute configuration for each labeled chiral center.")

    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.subheader(problem["name"].title())
        
        tab1, tab2 = st.tabs(["2D Numbered View", "3D Interactive View"])
        
        with tab1:
            st.image(molecule_image(mol), caption="Chiral carbons are explicitly labeled with their atom indices.")
            
        with tab2:
            with st.spinner("Generating 3D coordinates..."):
                try:
                    mol_block = get_3d_mol_block(problem["smiles"])
                    view = py3Dmol.view(width=450, height=350)
                    view.addModel(mol_block, 'mol')
                    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.35}})
                    
                    # Floating labels
                    for atom_idx in chiral_centers.keys():
                        view.addLabel(
                            f"C{atom_idx}", 
                            {
                                'fontColor': 'white', 
                                'backgroundColor': 'black', 
                                'fontSize': 14,
                                'backgroundOpacity': 0.8,
                                'alignment': 'center'
                            }, 
                            {'index': int(atom_idx)}
                        )
                        
                    view.zoomTo()
                    components.html(view._make_html(), height=350)
                    st.caption("Drag to rotate. Scroll to zoom. Find the lowest priority group!")
                except Exception as e:
                    st.error(f"Could not generate 3D model: {e}")

        st.markdown(f"*{problem.get('description', '')}*")
        
        if st.button("New Random Molecule", use_container_width=True, key="btn_new"):
            new_problem(generator)
            st.rerun()

    with right_col:
        st.subheader("Assign Absolute Configurations")
        
        if not chiral_centers:
            st.info("No chiral centers identified for this molecule.")
            return

        assignments: Dict[str, str] = {}
        
        with st.form("stereo_form"):
            for atom_idx in sorted(chiral_centers.keys(), key=int):
                assignments[atom_idx] = st.selectbox(
                    f"Carbon {atom_idx} (C{atom_idx}) Configuration",
                    ["Select R/S", "R", "S"],
                    key=f"stereo_{atom_idx}",
                )
            submitted = st.form_submit_button("Check Stereocenters", use_container_width=True)

        if submitted:
            incomplete = [f"C{idx}" for idx, choice in assignments.items() if choice == "Select R/S"]
            if incomplete:
                st.warning(f"Please assign a configuration for: {', '.join(incomplete)}")
            else:
                summary = ", ".join(f"C{idx} -> {choice}" for idx, choice in assignments.items())
                st.session_state.chat_history.append({"role": "user", "content": f"Submitted stereocenter assignments: {summary}."})

                with st.spinner("Asking the Socratic tutor..."):
                    try:
                        feedback = get_stereo_tutor_response(st.session_state.chat_history, problem, assignments)
                    except Exception as exc:
                        feedback = f"I could not reach the tutor API: {exc}"

                st.session_state.chat_history.append({"role": "assistant", "content": feedback})
                st.rerun()

    st.divider()
    st.subheader("Tutor Feedback")
    render_chat()

    follow_up = st.chat_input("Ask about CIP rules, Newman projections, or Chair conformations!")
    if follow_up:
        st.session_state.chat_history.append({"role": "user", "content": follow_up})
        with st.spinner("Thinking..."):
            try:
                feedback = get_stereo_tutor_response(st.session_state.chat_history, problem, {})
            except Exception as exc:
                feedback = f"I could not reach the tutor API: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": feedback})
        st.rerun()

if __name__ == "__main__":
    main()