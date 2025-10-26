import os
import json
import shlex

BASE_DIR = "E:\Ziyu\workspace\diff_aa_solution\pipeline\exp"  # your experiments root


class MiniTerminal:
    def __init__(self, base_dir):
        self.base_dir = os.path.abspath(base_dir)
        self.cwd = self.base_dir

    def run(self):
        print(f"MiniTerm started in {self.base_dir}")
        while True:
            try:
                cmd = input(f"{self._relative_cwd()}> ").strip()
                if not cmd:
                    continue
                args = shlex.split(cmd)
                command = args[0]
                if command == "exit":
                    print("Exiting MiniTerm...")
                    break
                elif command == "pwd":
                    print(self._relative_cwd())
                elif command == "cd":
                    self.cd(args[1] if len(args) > 1 else "")
                elif command == "ls":
                    self.ls()
                elif command == "info":
                    self.info(args[1] if len(args) > 1 else "")
                else:
                    print(f"Unknown command: {command}")
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print("Error:", e)

    def _relative_cwd(self):
        return os.path.relpath(self.cwd, self.base_dir)

    def cd(self, path):
        new_path = os.path.join(self.cwd, path)
        if not os.path.isdir(new_path):
            print("❌ Not a directory:", path)
            return
        self.cwd = os.path.abspath(new_path)

    def ls(self):
        for name in sorted(os.listdir(self.cwd)):
            full_path = os.path.join(self.cwd, name)
            if os.path.isdir(full_path):
                display_name = name
                meta_path = os.path.join(full_path, "@metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            display_name = data.get("id") or data.get("description") or name
                    except Exception:
                        pass
                print(f"[DIR] {display_name}")
            else:
                print(f"     {name}")

    def info(self, folder_name):
        """Show metadata for a subfolder"""
        full_path = os.path.join(self.cwd, folder_name)
        meta_path = os.path.join(full_path, "@metadata.json")
        if not os.path.exists(meta_path):
            print("No @metadata.json found.")
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(data, indent=4))


if __name__ == "__main__":
    term = MiniTerminal(BASE_DIR)
    term.run()
