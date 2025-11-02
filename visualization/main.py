import os
import subprocess
from pylatex import *
from pathlib import Path
from tqdm import tqdm
from fit.loss import ImageLoss, SmoothnessLoss, BandLoss, StraightnessLoss
import ipdb
import shutil
import random
import json

from fit import sh

def get_loss_summary(sub_path):
    """Extract loss information from loss_history.json files in subdirectory."""
    loss_info = {}
    
    # Check for loss history files in different training phases
    for phase in ['pass', 'pass_2', 'warmup']:
        json_path = sub_path / phase / 'loss_history.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                if data['iterations']:
                    # Get last 10 iterations data
                    num_iters = len(data['iterations'])
                    last_n_indices = list(range(max(0, num_iters - 10), num_iters))
                    
                    loss_info[phase] = {
                        'iterations': [data['iterations'][i] for i in last_n_indices],
                        'losses': {k: [v[i] for i in last_n_indices] for k, v in data['losses'].items()},
                        'differences': {k: [v[i] for i in last_n_indices] for k, v in data['loss_differences'].items()}
                    }
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not parse {json_path}: {e}")
                continue
    
    return loss_info

def format_loss_table(loss_info):
    """Format loss information as LaTeX table."""
    if not loss_info:
        return "No loss data available."
    
    # Start LaTeX table with better formatting for last 3 iterations
    table_latex = r"""
    \vspace{0.5em}
    \begin{table}[H]
    \centering
    \caption{Loss Values (Last 10 Iterations)}
    \tiny
    \begin{tabular}{|l|r|r|r|r|r|r|}
    \hline
    \textbf{Phase} & \textbf{Iteration} & \textbf{Image} & \textbf{Smooth} & \textbf{Band} & \textbf{Straightness} & \textbf{Total} \\
    \hline
    """
    
    # Add data rows for each phase and iteration
    for phase, data in loss_info.items():
        losses = data['losses']
        iterations = data['iterations']
        
        for i, iteration in enumerate(iterations):
            # First row for each phase shows the phase name, subsequent rows are empty
            phase_name = phase.title() if i == 0 else ""
            
            table_latex += f"{phase_name} & {iteration} & "
            table_latex += f"{losses.get('img_loss', [0])[i]:.4f} & "
            table_latex += f"{losses.get('smooth_loss', [0])[i]:.4f} & "
            table_latex += f"{losses.get('band_loss', [0])[i]:.4f} & "
            table_latex += f"{losses.get('straightness_loss', [0])[i]:.4f} & "
            table_latex += f"{losses.get('total_loss', [0])[i]:.4f} \\\\\n"
        
        # Add a separator line after each phase
        if len(loss_info) > 1:
            table_latex += r"\hline" + "\n"
    
    table_latex += r"""
    \end{tabular}
    \end{table}
    
    \begin{table}[H]
    \centering
    \caption{Loss Differences (Last 10 Iterations)}
    \tiny
    \begin{tabular}{|l|r|r|r|r|r|}
    \hline
    \textbf{Phase} & \textbf{Iteration} & \textbf{Image Diff} & \textbf{Smooth Diff} & \textbf{Band Diff} & \textbf{Straightness Diff} \\
    \hline
    """
    
    # Add difference rows for each phase and iteration
    for phase, data in loss_info.items():
        diffs = data['differences']
        iterations = data['iterations']
        
        for i, iteration in enumerate(iterations):
            # First row for each phase shows the phase name, subsequent rows are empty
            phase_name = phase.title() if i == 0 else ""
            
            table_latex += f"{phase_name} & {iteration} & "
            table_latex += f"{diffs.get('img_loss_diff', [0])[i]:.6f} & "
            table_latex += f"{diffs.get('smooth_loss_diff', [0])[i]:.6f} & "
            table_latex += f"{diffs.get('band_loss_diff', [0])[i]:.6f} & "
            table_latex += f"{diffs.get('straightness_loss_diff', [0])[i]:.6f} \\\\\n"
        
        # Add a separator line after each phase
        if len(loss_info) > 1:
            table_latex += r"\hline" + "\n"
    
    table_latex += r"""
    \end{tabular}
    \end{table}
    \vspace{0.5em}
    """
    
    return table_latex

def get_max_iter(sub_path):
    import re
    vis_path = sub_path / "vis"
    max_num = -1

    # regex to match files like iter_123
    pattern = re.compile(r"iter_(\d+)")

    for fname in os.listdir(vis_path):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num if max_num != -1 else None

def convert_svg_to_pdf(svg_path: Path) -> Path:
    """
    Convert an SVG file to PDF using Inkscape CLI.
    Returns the path to the generated PDF.
    """
    pdf_path = svg_path.with_suffix(".pdf")

    # Run Inkscape conversion (must be in PATH)
    subprocess.run([
        "inkscape",
        str(svg_path),
        "--export-type=pdf",
        f"--export-filename={pdf_path}"
    ], check=True)

    return pdf_path


# class MyDocument3Column(Document):
#     def __init__(self, output_path):
#         super().__init__(output_path, inputenc=None)

#         self.preamble.extend([
#             Command('title', 'Image Gallery'),
#             Command('author', 'Ziyu Sun'),
#         ])

#         self.packages.update([
#             Package('graphicx'),
#             Package('subcaption'),
#             Package('float')
#         ])

#         self.append(NoEscape(r"\maketitle"))

#         self.append(NoEscape(r"""
#         \section*{Illustration Guide}
#         \begin{itemize}
#             \item \textbf{Target}: Given raster image that we want to approximate with curves.
#             \item \textbf{Init Vector}: Initial vertices and edges.
#             \item \textbf{Init Render}: Rasterized image from init vector.
#             \item \textbf{Vector Result}: Final optimized vertices and edges.
#             \item \textbf{Render Result}: Rasterized image from vector result.
#             \item \textbf{Loss Curve}: Training loss evolution over iterations.
#         \end{itemize}
#         """))

#         self.append(NoEscape(r'\newpage'))

#         self.count = 0

#     def fill_document(self, image_path, doc):

#         for sub_path in tqdm(
#             sorted([p for p in image_path.iterdir() if p.is_dir()]),
#             desc="visualizing"
#         ):

#             # sub_path = sub_path / str(16)
#             if not sub_path.exists() or (sub_path / "render.png").exists() is False:
#                 continue

#             image_keys = ['target', "init vector", "init render", "vector result", "render result",\
#                            "loss curve", "segmentation", "2", "3", "4", "5", "6"]

#             image_paths = {
#                 key: sub_path / filename
#                 for key, filename in zip(image_keys, [
#                     'target.png',
#                     'init_vec.png',
#                     'init_render.png',
#                     'vec.png',
#                     'render.png',
#                     'loss.png',
#                     'contour.png',
#                     'contour.png',
#                     'contour.png',
#                     'contour.png',
#                     'contour.png',
#                     'contour.png',

#                 ])
#             }

#             with doc.create(Figure(position="H")) as images:
#                 for i, key in enumerate(image_keys):
#                     with doc.create(
#                         SubFigure(position="b", width=NoEscape(r"0.32\linewidth"))
#                     ) as subfig:

#                         img_path = image_paths[key]

#                         if img_path.suffix == '.svg':
#                             # Convert SVG → PDF once
#                             pdf_path = convert_svg_to_pdf(img_path)
#                             subfig.add_image(str(pdf_path), width=NoEscape(r"\linewidth"))
#                         else:
#                             subfig.add_image(str(img_path), width=NoEscape(r"\linewidth"))

#                         subfig.add_caption(key)

#                     if (i + 1) % 3 == 0:
#                         doc.append(NoEscape(r"\par\vspace{1em}"))

#                 name = sub_path.name
#                 images.add_caption(sub_path.name)

#                 self.count += 1
#                 if self.count % 3 == 0:
#                     self.append(NoEscape(r'\clearpage'))

class MyDocument4Column(Document):
    def __init__(self, output_path):
        super().__init__(output_path, inputenc=None)

        # Metadata
        self.preamble.extend([
            Command('title', 'Image Gallery'),
            Command('author', 'Ziyu Sun'),
        ])

        # Required packages
        self.packages.update([
            Package('graphicx'),
            Package('subcaption'),
            Package('float'),
            Package('indentfirst'),
            Package('array'),
            Package('booktabs'),
        ])

        # Title
        self.append(NoEscape(r"\maketitle"))


        self.append(NoEscape(r'\newpage'))
        self.append(NoEscape(r"\section*{Loss Function}"))
        self.append(NoEscape(r"\\[1em]"))
        self.append(NoEscape(ImageLoss().latex()))
        self.append(NoEscape(r"\\[1em]"))
        self.append(NoEscape(SmoothnessLoss(sh.pass_sh.smooth_loss).latex()))
        self.append(NoEscape(r"\\[1em]"))
        self.append(NoEscape(BandLoss(sh.pass_sh.band_loss).latex()))
        self.append(NoEscape(r"\\[1em]"))
        self.append(NoEscape(StraightnessLoss(sh.pass_sh.straightness_loss).latex()))

        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path: Path, doc: Document):

        # Get all subdirectories and shuffle them randomly
        all_subdirs = [p for p in image_path.iterdir() if p.is_dir()]
        
        for sub_path in tqdm(
            all_subdirs,
            desc="visualizing"
        ):

            if not sub_path.exists() or (sub_path / "vec.png").exists() is False:
                continue

            image_keys = [
                'target', "contour", "contour udf", "init vec",\
                "vec", "vec bg", "loss"
            ]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'contour.png',
                    "sdf.png",
                    'init.png',
                    'vec.png',
                    'vec_bg.png',
                    'loss_pass.png'
                ])
            }

            # 4-column figure
            with doc.create(Figure(position="H")) as images:
                for i, key in enumerate(image_keys):
                    # first page
                    with doc.create(
                        SubFigure(position="b", width=NoEscape(r"0.24\linewidth"))
                    ) as subfig:

                        img_path = image_paths[key]

                        if img_path.suffix == '.svg':
                            # Convert SVG → PDF if needed
                            pdf_path = convert_svg_to_pdf(img_path)
                            subfig.add_image(str(pdf_path), width=NoEscape(r"\linewidth"))
                        else:
                            subfig.add_image(str(img_path), width=NoEscape(r"\linewidth"))

                        subfig.add_caption(key)

                    # Break line after 4 subfigures
                    if (i + 1) % 4 == 0:
                        doc.append(NoEscape(r"\par\vspace{1em}"))
            
            # Add loss information section
            loss_info = get_loss_summary(sub_path)
            if loss_info:
                doc.append(NoEscape(r"\section*{Loss Summary for " + sub_path.name + "}"))
                loss_table = format_loss_table(loss_info)
                doc.append(NoEscape(loss_table))
            else:
                doc.append(NoEscape(r"\section*{Loss Summary for " + sub_path.name + "}"))
                doc.append(NoEscape(r"No loss data available for this experiment."))
            
                # next page
            doc.append(NoEscape(r'\newpage'))






def run_latex(image_path, output_path, delete_vis=False):
    doc = MyDocument4Column(output_path)
    doc.fill_document(image_path, doc)
    try:
        doc.generate_pdf(
            output_path,
            clean_tex=False,
            compiler='pdflatex',
            compiler_args=["-interaction=nonstopmode"]
        )
    except subprocess.CalledProcessError:
        print("⚠️ LaTeX returned non-zero exit code, but PDF should still exist.")



if __name__ == "__main__":

    exp_path = Path(r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\10-31\23-10-37")
    output_path = exp_path / "res"
    image_path = exp_path
    run_latex(image_path, output_path, delete_vis=False)