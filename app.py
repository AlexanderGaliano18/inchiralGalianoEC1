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

# Configuración para evitar warnings de RDKit
import warnings
warnings.filterwarnings('ignore')

# Función para instalar paquetes
def install_package(package):
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# Manejo de importación de RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("❌ RDKit no está instalado. Por favor instala RDKit para usar la funcionalidad completa.")
    st.info("Instala con: pip install rdkit")
    RDKIT_AVAILABLE = False

# Manejo de importación de stmol para visualización 3D
try:
    import stmol
    import py3Dmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

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
            if i + 1 < len(smiles) and smiles[i+1] == "@":
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
                posiciones.append((i, True))  # ya es @@
                i += 2
            else:
                posiciones.append((i, False))  # es @ simple
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
    """Genera imagen 2D de una molécula usando RDKit"""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generar coordenadas 2D si no las tiene
        AllChem.Compute2DCoords(mol)
        
        # Crear imagen
        img = Draw.MolToImage(mol, size=(width, height))
        return img
        
    except Exception as e:
        st.error(f"Error generando imagen 2D: {str(e)}")
        return None

def generar_grid_2d(smiles_list, mols_per_row=4, mol_size=(200, 200)):
    """Genera una grilla de moléculas 2D"""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mols = []
        legends = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                legends.append(f"Isómero {i+1}")
        
        if not mols:
            return None
            
        img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=mols_per_row,
            subImgSize=mol_size,
            legends=legends
        )
        return img
        
    except Exception as e:
        st.error(f"Error generando grilla 2D: {str(e)}")
        return None

def mol_to_3d_block(smiles):
    """Convierte SMILES a bloque MOL 3D para visualización"""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Generar conformación 3D
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result != 0:
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result != 0:
                return None
        
        # Optimización con campo de fuerza
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass
        
        # Convertir a bloque MOL
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
        
    except Exception as e:
        return None

def render_3d_molecule(mol_block, style='stick'):
    """Renderiza molécula 3D usando stmol"""
    if not STMOL_AVAILABLE:
        st.error("❌ Librerías de visualización 3D no disponibles")
        return
    
    try:
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mol_block, 'mol')
        view.setStyle({style: {}})
        view.zoomTo()
        
        stmol.showmol(view, height=400, width=400)
        
    except Exception as e:
        st.error(f"Error renderizando molécula 3D: {str(e)}")

def smiles_to_xyz(smiles, mol_id):
    if not RDKIT_AVAILABLE:
        return None, "❌ RDKit no está disponible"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"❌ Error: SMILES inválido {smiles}"
        
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  
        
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result != 0:
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result != 0:
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
        - 🎨 Visualización 2D de moléculas individuales
        - 📊 Grillas comparativas de isómeros
        - 🌐 Visualización 3D interactiva
        - 💾 Exportación a XYZ
        
        **Ejemplos de SMILES:**
        - Etanol: `CCO`
        - Molécula quiral: `CC(O)C(N)C`
        - Con quiralidad: `C[C@H](O)[C@@H](N)C`
        - Aminoácido: `N[C@@H](C)C(=O)O`
        """)
        
        # Estado de librerías
        st.markdown("---")
        st.subheader("📦 Estado de Librerías")
        if RDKIT_AVAILABLE:
            st.success("✅ RDKit: Disponible")
        else:
            st.error("❌ RDKit: No disponible")
            
        if STMOL_AVAILABLE:
            st.success("✅ Visualización 3D: Habilitada")
        else:
            st.warning("⚠️ Visualización 3D: No disponible")
            st.info("Para habilitar, instala: `pip install stmol py3Dmol`")
    
    st.subheader("📝 Entrada de Datos")
    smiles_input = st.text_input(
        "👉 Ingresa el código SMILES:",
        placeholder="Ejemplo: C[C@H](O)[C@@H](N)C"
    )
    
    if smiles_input:
        # Visualización 2D de la molécula original
        if RDKIT_AVAILABLE:
            st.subheader("🎨 Visualización 2D - Molécula Original")
            img_2d = generar_imagen_2d(smiles_input)
            if img_2d:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(img_2d, caption=f"Estructura 2D: {smiles_input}", use_column_width=True)
            else:
                st.error("❌ No se pudo generar la imagen 2D")
        
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
                            tipo_quiralidad = str(chirality) if chirality else "Sin asignar"
                            st.write(f"• Átomo {idx}: {tipo_quiralidad}")
                else:
                    if "inválido" in mensaje_quiralidad:
                        st.error(f"❌ {mensaje_quiralidad}")
                    else:
                        st.warning(f"⚠️ {mensaje_quiralidad}")
            else:
                st.warning("⚠️ RDKit no disponible para análisis")
        
        with col2:
            st.info(f"**📋 Centros especificados en SMILES:**")
            if centros_especificados > 0:
                st.success(f"✅ {centros_especificados} centros con @ o @@ especificados")
                for pos in posiciones_at:
                    st.write(f"• Posición {pos}")
            else:
                st.warning("⚠️ No hay centros especificados con @ o @@")
        
        if RDKIT_AVAILABLE and es_quiral and centros_especificados == 0:
            st.info("""
            💡 Tu molécula es quiral pero no tiene centros especificados con @ o @@.
            Ejemplo: `CC(O)C(N)C` → `C[C@H](O)[C@@H](N)C`
            """)
        
        # Generar estereoisómeros
        isomeros, n_centros = [], 0
        if centros_especificados > 0:
            with st.spinner("🔄 Generando estereoisómeros..."):
                isomeros, n_centros = generar_estereoisomeros(smiles_input)
        
        # Visualización 3D de la molécula original
        if STMOL_AVAILABLE and RDKIT_AVAILABLE:
            st.subheader("🌐 Visualización 3D - Molécula Original")
            mol_block = mol_to_3d_block(smiles_input)
            if mol_block:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Opciones de visualización:**")
                    style_3d = st.selectbox(
                        "Estilo 3D:",
                        options=['stick', 'sphere', 'line', 'cartoon'],
                        index=0
                    )
                with col2:
                    render_3d_molecule(mol_block, style_3d)
            else:
                st.error("❌ No se pudo generar estructura 3D")
        
        # Crear tabs
        if isomeros:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Lista", "🎨 Visualización 2D", "🌐 Visualización 3D", "💾 Descargar SMI", "🧪 Convertir XYZ"
            ])
            
            with tab1:
                col1, col2 = st.columns(2)
                for i, isomero in enumerate(isomeros):
                    if i % 2 == 0:
                        col1.code(f"{i+1}. {isomero}")
                    else:
                        col2.code(f"{i+1}. {isomero}")
            
            with tab2:
                if RDKIT_AVAILABLE:
                    st.subheader("🎨 Grilla de Estereoisómeros 2D")
                    
                    # Opciones de visualización
                    col1, col2 = st.columns(2)
                    with col1:
                        mols_per_row = st.slider("Moléculas por fila:", 2, 6, 4)
                    with col2:
                        mol_size = st.slider("Tamaño de moléculas:", 150, 300, 200)
                    
                    if st.button("🖼️ Generar Grilla 2D", type="primary"):
                        with st.spinner("Generando grilla..."):
                            grid_img = generar_grid_2d(
                                isomeros, 
                                mols_per_row=mols_per_row, 
                                mol_size=(mol_size, mol_size)
                            )
                            if grid_img:
                                st.image(grid_img, caption="Comparación de todos los estereoisómeros")
                                
                                # Botón para descargar imagen
                                buf = io.BytesIO()
                                grid_img.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 Descargar Grilla PNG",
                                    data=buf.getvalue(),
                                    file_name="estereoisomeros_2d.png",
                                    mime="image/png"
                                )
                            else:
                                st.error("❌ Error generando grilla 2D")
                    
                    # Visualización individual
                    st.subheader("🔍 Visualización Individual")
                    selected_idx = st.selectbox(
                        "Selecciona un estereoisómero:",
                        range(len(isomeros)),
                        format_func=lambda x: f"Isómero {x+1}: {isomeros[x]}"
                    )
                    
                    img_individual = generar_imagen_2d(isomeros[selected_idx], width=400, height=400)
                    if img_individual:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(
                                img_individual, 
                                caption=f"Isómero {selected_idx+1}: {isomeros[selected_idx]}"
                            )
                else:
                    st.warning("⚠️ RDKit requerido para visualización 2D")
            
            with tab3:
                if STMOL_AVAILABLE and RDKIT_AVAILABLE:
                    st.subheader("🌐 Visualización 3D Interactiva")
                    
                    # Selección de isómero
                    selected_3d_idx = st.selectbox(
                        "Selecciona estereoisómero para 3D:",
                        range(len(isomeros)),
                        format_func=lambda x: f"Isómero {x+1}: {isomeros[x]}",
                        key="3d_selector"
                    )
                    
                    # Opciones de estilo
                    col1, col2 = st.columns(2)
                    with col1:
                        style_option = st.selectbox(
                            "Estilo de visualización:",
                            options=['stick', 'sphere', 'line', 'cartoon'],
                            index=0,
                            key="3d_style"
                        )
                    
                    selected_smiles = isomeros[selected_3d_idx]
                    mol_block_3d = mol_to_3d_block(selected_smiles)
                    
                    if mol_block_3d:
                        st.markdown(f"**Visualizando:** Isómero {selected_3d_idx+1} - `{selected_smiles}`")
                        render_3d_molecule(mol_block_3d, style_option)
                    else:
                        st.error(f"❌ No se pudo generar estructura 3D para el isómero {selected_3d_idx+1}")
                        
                else:
                    if not STMOL_AVAILABLE:
                        st.info("🔄 Instalando librerías de visualización 3D...")
                    if not RDKIT_AVAILABLE:
                        st.warning("⚠️ RDKit requerido para visualización 3D")
            
            with tab4:
                smi_content = "\n".join(isomeros)
                st.download_button(
                    label="📥 Descargar archivo.smi",
                    data=smi_content,
                    file_name="estereoisomeros.smi",
                    mime="text/plain"
                )
                with st.expander("👀 Vista previa del archivo SMI"):
                    st.text(smi_content)
            
            with tab5:
                if st.button("🚀 Convertir todos a XYZ", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    archivos_xyz = {}
                    mensajes_log = []
                    
                    for i, smiles in enumerate(isomeros):
                        try:
                            progress = (i + 1) / len(isomeros)
                            progress_bar.progress(progress)
                            status_text.text(f"Procesando molécula {i+1}/{len(isomeros)}: {smiles}")
                            
                            xyz_content, mensaje = smiles_to_xyz(smiles, i+1)
                            mensajes_log.append(mensaje)
                            
                            if xyz_content:
                                archivos_xyz[f"mol_{i+1}.xyz"] = xyz_content
                        except Exception as e:
                            mensajes_log.append(f"❌ Error procesando molécula {i+1}: {str(e)}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Proceso completado!")
                    
                    with st.expander("📋 Log de procesamiento"):
                        for mensaje in mensajes_log:
                            if "❌" in mensaje or "⚠️" in mensaje:
                                st.error(mensaje)
                            else:
                                st.success(mensaje)
                    
                    if archivos_xyz:
                        zip_data = crear_archivo_zip(archivos_xyz)
                        st.download_button(
                            label="📦 Descargar archivos XYZ (ZIP)",
                            data=zip_data,
                            file_name="estereoisomeros_xyz.zip",
                            mime="application/zip"
                        )
                        with st.expander("👀 Vista previa del primer archivo XYZ"):
                            primer_archivo = list(archivos_xyz.values())[0]
                            st.code(primer_archivo)
        else:
            st.info("💡 Ingresa un SMILES con centros quirales especificados (@ o @@) para generar estereoisómeros")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <small>🧬 <strong>Inchiral Enhanced</strong> - Universidad Científica del Sur<br>
            Generador y Visualizador de Estereoisómeros | RDKit + Streamlit + stmol</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
