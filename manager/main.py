from pylatex import Document, Figure, NoEscape, NewPage
import os

# === Paths ===
base_dir = r"E:\Ziyu\workspace\diff_aa_solution\pipeline\exp\10-08"
output_path = r"E:\Ziyu\workspace\diff_aa_solution\pipeline\manager\compare_five"

# === Folder discovery ===
main_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
subfolders = sorted([
    f for f in os.listdir(os.path.join(base_dir, main_folders[0]))
    if os.path.isdir(os.path.join(base_dir, main_folders[0], f))
])

# === LaTeX document setup ===
doc = Document("vec_images", documentclass="article")
doc.preamble.append(NoEscape(r'\usepackage[margin=1in]{geometry}'))
doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))
doc.preamble.append(NoEscape(r'\usepackage{subcaption}'))
doc.preamble.append(NoEscape(r'\usepackage{float}'))

# === Loop through each subfolder (each page) ===
for sub in subfolders:
    # Escape underscores for LaTeX section title

    first_folder = main_folders[0]
    aa_path = os.path.join(base_dir, first_folder, sub, "aa_64.png")

    # Skip this subfolder entirely if aa_64.png does not exist
    if not os.path.exists(aa_path):
        print(f"Skipping '{sub}' — aa_64.png not found.")
        continue

    safe_sub = sub.replace('_', r'\_')
    doc.append(NoEscape(r'\section*{%s}' % safe_sub))

    # --- First row: aa_64.png, init_vec.png, first 2 vec.png ---
    with doc.create(Figure(position='H')) as fig:
        fig.append(NoEscape(r'\centering'))

        first_folder = main_folders[0]
        aa_path = os.path.join(base_dir, first_folder, sub, "aa_64.png")
        init_path = os.path.join(base_dir, first_folder, sub, "init_vec.png")

        # Add the two initial reference images
        for label, path in [("aa_64", aa_path), ("init_vec", init_path)]:
            if os.path.exists(path):
                safe_label = label.replace('_', r'\_')
                fig.append(NoEscape(r'\begin{subfigure}{0.23\textwidth}'))
                fig.append(NoEscape(r'\includegraphics[width=\linewidth]{%s}' % path.replace('\\', '/')))
                fig.append(NoEscape(r'\caption*{%s}' % safe_label))
                fig.append(NoEscape(r'\end{subfigure}'))

        # Add first 2 vec.png images
        for folder in main_folders[:2]:
            img_path = os.path.join(base_dir, folder, sub, "vec.png")
            if os.path.exists(img_path):
                safe_folder = folder.replace('_', r'\_')
                fig.append(NoEscape(r'\begin{subfigure}{0.23\textwidth}'))
                fig.append(NoEscape(r'\includegraphics[width=\linewidth]{%s}' % img_path.replace('\\', '/')))
                fig.append(NoEscape(r'\caption*{%s}' % safe_folder))
                fig.append(NoEscape(r'\end{subfigure}'))

    # --- Second row: remaining vec.png images ---
    with doc.create(Figure(position='H')) as fig2:
        fig2.append(NoEscape(r'\centering'))
        for folder in main_folders[2:]:
            img_path = os.path.join(base_dir, folder, sub, "vec.png")
            if os.path.exists(img_path):
                safe_folder = folder.replace('_', r'\_')
                fig2.append(NoEscape(r'\begin{subfigure}{0.3\textwidth}'))
                fig2.append(NoEscape(r'\includegraphics[width=\linewidth]{%s}' % img_path.replace('\\', '/')))
                fig2.append(NoEscape(r'\caption*{%s}' % safe_folder))
                fig2.append(NoEscape(r'\end{subfigure}'))

    # --- Page break for next subfolder ---
    doc.append(NewPage())

# === Generate PDF ===
doc.generate_pdf(output_path, clean_tex=False)
