import os


def rename_files_recursively(folder_path):
    # Verificar se a pasta existe
    if not os.path.isdir(folder_path):
        print(f"Pasta {folder_path} não encontrada.")
        return

    # Percorrer todas as pastas e subpastas
    for root, _, files in os.walk(folder_path):
        # Obter o nome da pasta atual
        folder_name = os.path.basename(root)

        for filename in files:
            # Obter o caminho completo do arquivo
            file_path = os.path.join(root, filename)

            # Verificar se é um arquivo (os.walk() já filtra diretórios, mas essa verificação adicional é segura)
            if os.path.isfile(file_path):
                # Obter o nome e a extensão do arquivo
                name, ext = os.path.splitext(filename)

                # Criar o novo nome de arquivo
                new_name = f"{name}_{folder_name}{ext}"
                new_file_path = os.path.join(root, new_name)

                # Renomear o arquivo
                os.rename(file_path, new_file_path)
                print(f"Renomeado: {file_path} -> {new_file_path}")


# Definir o caminho da pasta principal
folder_path = 'Dataset'

# Rodar a função
rename_files_recursively(folder_path)
