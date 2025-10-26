import os
import subprocess
from pylatex import *
from pathlib import Path
from tqdm import tqdm
from fit.loss import ImageLoss, SmoothnessLoss, BandLoss, StraightnessLoss
import ipdb
import shutil
import random

from fit import sh

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
        ])

        # Title
        self.append(NoEscape(r"\maketitle"))


        # self.append(NoEscape(r'\newpage'))
        # self.append(NoEscape(r"\section*{Loss Function}"))
        # self.append(NoEscape(r"\\[1em]"))
        # self.append(NoEscape(ImageLoss().latex()))
        # self.append(NoEscape(r"\\[1em]"))
        # self.append(NoEscape(SmoothnessLoss().latex()))
        # self.append(NoEscape(r"\\[1em]"))
        # self.append(NoEscape(BandLoss().latex_1()))
        # self.append(NoEscape(r"\\[1em]"))
        # self.append(NoEscape(StraightnessLoss().latex()))

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
                'target', "contour", "contour udf", "init vec", "vec warmup", "vec 1 pass", "vec 2 pass",\
                "vec", "vec bg"
            ]

            image_paths = {
                key: sub_path / filename
                for key, filename in zip(image_keys, [
                    'target.png',
                    'contour.png',
                    "sdf.png",
                    'init_vec.png',
                    'vec_warmup.png',
                    'vec_pass.png',
                    'vec_pass_2.png',
                    'vec.png',
                    'vec_bg.png'
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
                
                # next page
            doc.append(NoEscape(r'\newpage'))

            # Add points visualization page
            add_points_path = sub_path / "add_points"
            if add_points_path.exists():
                # Find all add_points_iter_*.png files
                add_points_files = sorted(add_points_path.glob("add_points_iter_*.png"))
                
                if add_points_files:
                    doc.append(NoEscape(r"\section*{Add Points Iterations}"))
                    
                    # Process images in batches of 20 per page
                    images_per_page = 20
                    for page_start in range(0, len(add_points_files), images_per_page):
                        page_files = add_points_files[page_start:page_start + images_per_page]
                        
                        with doc.create(Figure(position="H")) as add_fig:
                            for i, img_file in enumerate(page_files):
                                # Extract iteration number from filename (add_points_iter_X.png)
                                try:
                                    iter_num = img_file.stem.split('_')[-1]  # Gets the X from add_points_iter_X
                                except:
                                    iter_num = str(page_start + i)  # Fallback to global index if parsing fails
                                
                                with doc.create(
                                    SubFigure(position="b", width=NoEscape(r"0.24\linewidth"))
                                ) as subfig:
                                    subfig.add_image(str(img_file), width=NoEscape(r"\linewidth"))
                                    subfig.add_caption(f"midpoint {iter_num}")

                                # Break line after 4 subfigures
                                if (i + 1) % 4 == 0:
                                    doc.append(NoEscape(r"\par\vspace{1em}"))
                        
                        # Add page break if there are more images to show
                        if page_start + images_per_page < len(add_points_files):
                            doc.append(NoEscape(r'\newpage'))
                    
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

    exp_path = Path(r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\10-22\22-03-14")
    output_path = exp_path / "res"
    image_path = exp_path
    run_latex(image_path, output_path, delete_vis=False)
