import json
import os
import re

notebooks_dir = "C:/Users/omjee/.gemini/antigravity/scratch/ecommerce-customer-behavior-eda/notebooks"
files = [
    "00_generate_data.py",
    "01_data_cleaning.py",
    "02_eda_analysis.py",
    "03_customer_segmentation.py",
    "04_insights_summary.py"
]

setup_source = [
    "# Clone repository and setup\n",
    "!git clone https://github.com/amitkumar227/ecommerce-customer-behavior-eda.git\n",
    "import os\n",
    "os.chdir('ecommerce-customer-behavior-eda')\n",
    "!pip install -q pandas numpy matplotlib seaborn scipy"
]

def create_notebook(filename):
    filepath = os.path.join(notebooks_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # fix paths
    content = content.replace("../data", "data").replace("../outputs", "outputs")
    
    # extract header docstring
    header_match = re.match(r'\"\"\"(.*?)\"\"\"', content, re.DOTALL)
    title_desc = ""
    if header_match:
        title_desc = header_match.group(1).strip()
        content = content[header_match.end():].strip()
    
    cells = []
    
    # 1. Title cell
    if title_desc:
        lines = title_desc.split('\n')
        title_source = []
        for i, line in enumerate(lines):
            if i == 0:
                title_source.append(f"# {line.strip()}\n")
            else:
                s = line.strip() + ("\n" if i < len(lines)-1 else "")
                title_source.append(s)
                
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": title_source
        })
        
    # 2. Setup cell
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": setup_source
    })
    
    # Split code by looking for blocks that start with `# NUMBER.` or `# ----`
    # We will just split by double newline and check if the block looks like a new section
    
    blocks = re.split(r'\n\n(?=# )', content)
    for i, block in enumerate(blocks):
        if i > 0:
            block = "# " + block
            
        lines = block.split('\n')
        header = ""
        # Identify if it's a section
        if lines[0].startswith('# ') and len(lines[0]) > 3:
            h = lines[0].replace('#', '').replace('-', '').strip()
            if any(c.isupper() or c.isdigit() for c in h):
                header = h
                
        if header:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {header}"]
            })
            
        source_lines = [l + '\n' for l in lines]
        if source_lines:
            source_lines[-1] = source_lines[-1].rstrip('\n')
        
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source_lines
        })
            
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": [],
                "toc_visible": True
            }
        },
        "cells": cells
    }
    
    out_name = filename.replace('.py', '.ipynb')
    out_path = os.path.join(notebooks_dir, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

for f in files:
    create_notebook(f)
