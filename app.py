import streamlit as st
import itertools
import os
import zipfile
import tempfile
import io
import sys
import subprocess
from PIL import Image
import base64
import warnings
warnings.filterwarnings('ignore')

# Función para instalar paquetes
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("❌ RDKit no está instalado. Instala con: pip install rdkit")
    RDKIT_AVAILABLE = False

# py3Dmol para 3D
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    install_package("py3Dmol")
    import py3Dmol
    PY3DMOL_AVAILABLE = True

# ---------------- Funciones ---------------- #

def detectar_quiralidad(smiles: str):
    if not RDKIT_AVAILABLE:
        return False, "RDKit no disponible", []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "SMILES inválido", []
        centros = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(centros) == 0:
            return False, "Su molécula no es quiral", []
        else:
            return True, f"Su molécula es quiral. Se detectaron {len(centros)} posibles centros", centros
    except Exception as e:
        return False, f"Error al analizar la molécula: {str(e)}", []

def analizar_centros_existentes(smiles: str):
    centros_especificados = 0
    posiciones_at = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "@":
            if i+1 < len(smiles) and smiles[i+1] == "@":
                centros_especificados += 1
                posiciones_at.append(i)
                i += 2
            else:
                centros_especificados += 1
                posiciones_at.append(i)
                i += 1
        else:
            i += 1
    return centros_especificados, posiciones_at

def generar_estereoisomeros(smiles: str):
    posiciones = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "@":
            if i + 1 < len(smiles) and smiles[i+1] == "@":
                posiciones.append((i, True))
                i += 2
            else:
                posiciones.append((i, False))
                i += 1
        else:
            i += 1
    n = len(posiciones)
    if n == 0:
        st.warning("⚠️ El SMILES no tiene centros quirales especificados con @ o @@. No se generarán isómeros.")
        return [], n
    elif n > 3:
        st.error("❌ El SMILES tiene más de 3 centros quirales. No se generarán isómeros.")
        return [], n
    combinaciones = list(itertools.product(["@", "@@"], repeat=n))
    resultados = []
    for comb in combinaciones:
        chars = list(smiles)
        offset = 0
        for (pos, era_doble), val in zip(posiciones, comb):
            real_pos = pos + offset
            if era_doble:
                chars[real_pos:real_pos+2] = list(val)
                offset += len(val) - 2
            else:
                chars[real_pos:real_pos+1] = list(val)
                offset += len(val) - 1
        resultados.append("".join(chars))
    return resultados, n

def generar_imagen_2d(smiles, width=300, height=300):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(width, height))
        return img
    except:
        return None

def generar_grid_2d(smiles_list, mols_per_row=4, mol_size=(200, 200)):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mols = []
        legends = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                legends.append(f"Isómero {i+1}")
        if not mols:
            return None
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=mol_size, legends=legends)
        return img
    except:
        return None

def smiles_to_xyz(smiles, mol_id):
    if not RDKIT_AVAILABLE:
        return None, "❌ RDKit no disponible"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"❌ SMILES inválido {smiles}"
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None, f"⚠️ No se pudo generar conformación 3D para {smiles}"
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass
        conf = mol.GetConformer()
        xyz_content = f"{mol.GetNumAtoms()}\n{smiles}\n"
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_content += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
        return xyz_content, f"✅ Molécula {mol_id} procesada correctamente"
    except Exception as e:
        return None, f"❌ Error procesando {smiles}: {str(e)}"

def crear_archivo_zip(archivos_xyz):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in archivos_xyz.items():
            zip_file.writestr(filename, content)
    return zip_buffer.getvalue()

# ---------------- Función 3D ---------------- #
def mostrar_molecula_3d(xyz_content, width=400, height=400):
    if not PY3DMOL_AVAILABLE:
        st.warning("⚠️ py3Dmol no disponible")
        return
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_content, "xyz")
    view.setStyle({'stick': {}})
    view.zoomTo()
    html_view = view._make_html()
    st.components.v1.html(html_view, height=height)

# ---------------- Main App ---------------- #
def main():
    st.set_page_config(
        page_title="Inchiral - Generador de Estereoisómeros",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 Generador de Estereoisómeros con Visualización")
    st.markdown("**Genera estereoisómeros y visualízalos en 2D y 3D**")
    
    with st.sidebar:
        try:
            st.image("imagenes/Inchiral.png", width=200)
        except:
            st.markdown("**🧬 Inchiral**")
        st.markdown("---")
        st.title("ℹ️ Información")
        st.markdown("""
        **Funcionalidades:**
        - 🔍 Detección automática de quiralidad
        - 🧪 Generación de todos los estereoisómeros
        - 🎨 Visualización 2D
        - 🔬 Visualización 3D
        - 💾 Exportación a XYZ
        """)
        st.markdown("---")
        st.subheader("📦 Estado de Librerías")
        if RDKIT_AVAILABLE:
            st.success("✅ RDKit: Disponible")
        else:
            st.error("❌ RDKit: No disponible")
        if PY3DMOL_AVAILABLE:
            st.success("✅ py3Dmol: Disponible")
        else:
            st.warning("⚠️ py3Dmol: No disponible")
    
    st.subheader("📝 Entrada de Datos")
    smiles_input = st.text_input(
        "👉 Ingresa el código SMILES:",
        placeholder="Ejemplo: C[C@H](O)[C@@H](N)C"
    )
    
    if smiles_input:
        if RDKIT_AVAILABLE:
            st.subheader("🎨 Visualización 2D - Molécula Original")
            img_2d = generar_imagen_2d(smiles_input)
            if img_2d:
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.image(img_2d, caption=f"Estructura 2D: {smiles_input}", use_container_width=True)
        
        st.subheader("🔍 Análisis de Quiralidad")
        es_quiral, mensaje_quiralidad, centros_detectados = detectar_quiralidad(smiles_input)
        centros_especificados, posiciones_at = analizar_centros_existentes(smiles_input)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**🔎 Análisis con RDKit:**")
            if RDKIT_AVAILABLE:
                if es_quiral:
                    st.success(f"✅ {mensaje_quiralidad}")
                    if centros_detectados:
                        st.write("**Centros detectados:**")
                        for i, (idx, chirality) in enumerate(centros_detectados):
                            tipo = str(chirality) if chirality else "Sin asignar"
                            st.write(f"• Átomo {idx}: {tipo}")
                else:
                    st.warning(f"⚠️ {mensaje_quiralidad}")
        with col2:
            st.info(f"**📋 Centros especificados en SMILES:**")
            if centros_especificados>0:
                st.success(f"✅ {centros_especificados} centros con @ o @@ especificados")
                for pos in posiciones_at:
                    st.write(f"• Posición {pos}")
            else:
                st.warning("⚠️ No hay centros especificados con @ o @@")
        
        if RDKIT_AVAILABLE and es_quiral and centros_especificados==0:
            st.info("💡 Tu molécula es quiral pero no tiene centros especificados con @ o @@")
        
        isomeros, n_centros = [],0
        if centros_especificados>0:
            with st.spinner("🔄 Generando estereoisómeros..."):
                isomeros, n_centros = generar_estereoisomeros(smiles_input)
        
        if isomeros:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Lista", "🎨 2D", "💾 Descargar SMI", "🧪 Convertir XYZ", "🔬 3D"
            ])
            
            with tab1:
                col1, col2 = st.columns(2)
                for i, iso in enumerate(isomeros):
                    if i%2==0:
                        col1.code(f"{i+1}. {iso}")
                    else:
                        col2.code(f"{i+1}. {iso}")
            
            with tab2:
                if RDKIT_AVAILABLE:
                    st.subheader("🎨 Grilla 2D")
                    col1, col2 = st.columns(2)
                    with col1:
                        mols_per_row = st.slider("Moléculas por fila:", 2,6,4)
                    with col2:
                        mol_size = st.slider("Tamaño de moléculas:", 150,300,200)
                    if st.button("🖼️ Generar Grilla 2D", type="primary"):
                        with st.spinner("Generando grilla..."):
                            grid_img = generar_grid_2d(isomeros, mols_per_row, (mol_size, mol_size))
                            if grid_img:
                                st.image(grid_img, caption="Comparación de todos los estereoisómeros")
                                buf = io.BytesIO()
                                grid_img.save(buf, format='PNG')
                                st.download_button("📥 Descargar Grilla PNG", buf.getvalue(), "estereoisomeros_2d.png", "image/png")
            
            with tab3:
                smi_content = "\n".join(isomeros)
                st.download_button("📥 Descargar archivo.smi", smi_content, "estereoisomeros.smi", "text/plain")
                with st.expander("👀 Vista previa"):
                    st.text(smi_content)
            
            with tab4:
                if st.button("🚀 Convertir todos a XYZ", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    archivos_xyz = {}
                    mensajes_log = []
                    for i, smiles in enumerate(isomeros):
                        progress = (i+1)/len(isomeros)
                        progress_bar.progress(progress)
                        status_text.text(f"Procesando molécula {i+1}/{len(isomeros)}")
                        xyz_content, mensaje = smiles_to_xyz(smiles, i+1)
                        mensajes_log.append(mensaje)
                        if xyz_content:
                            archivos_xyz[f"mol_{i+1}.xyz"] = xyz_content
                    progress_bar.progress(1.0)
                    status_text.text("✅ Proceso completado")
                    with st.expander("📋 Log de procesamiento"):
                        for m in mensajes_log:
                            if "❌" in m or "⚠️" in m:
                                st.error(m)
                            else:
                                st.success(m)
                    if archivos_xyz:
                        zip_data = crear_archivo_zip(archivos_xyz)
                        st.download_button("📦 Descargar archivos XYZ (ZIP)", zip_data, "estereoisomeros_xyz.zip", "application/zip")
            
            with tab5:
                st.subheader("🔬 Visualización 3D")
                if PY3DMOL_AVAILABLE:
                    selected_idx = st.selectbox("Selecciona un isómero", range(len(isomeros)), format_func=lambda x: f"Isómero {x+1}")
                    xyz_content, mensaje = smiles_to_xyz(isomeros[selected_idx], selected_idx+1)
                    if xyz_content:
                        mostrar_molecula_3d(xyz_content, 400, 400)
                else:
                    st.warning("⚠️ py3Dmol no disponible para 3D")
    
    st.markdown("---")
    st.markdown("<div style='text-align:center'><small>🧬 Inchiral Enhanced - UCS</small></div>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
