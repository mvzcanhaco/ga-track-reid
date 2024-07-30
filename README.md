# Projeto de Re-Identificação de Pessoas com Algoritmo Genético

Este projeto tem como objetivo otimizar os parâmetros para a detecção e rastreamento de pessoas em vídeos e imagens, utilizando um algoritmo genético. O projeto utiliza modelos de detecção (YOLOv8), extração de características (MobileNetV2), e rastreamento (DeepSort) para maximizar a precisão da detecção e re-identificação de indivíduos.

## Estrutura do Projeto

- `main.py`: Script principal que inicia o processo de otimização genética e coordena as funções de avaliação.
- `genetic_algorithm.py`: Implementação do algoritmo genético, incluindo inicialização da população, crossover, mutação e seleção.
- `processing_functions.py`: Funções para processamento de frames, extração de características, e avaliação de métricas.
- `visualization.py`: Funções para visualização do progresso, incluindo a exibição de frames processados e gráficos de desempenho.
- `requirements.txt`: Lista de dependências do projeto.

## Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas e ferramentas:

- Python 3.9 ou superior
- As bibliotecas listadas no arquivo `requirements.txt`

## Instalação

1. **Clone o repositório:** Clone o repositório do projeto em seu ambiente local.
```bash
git clone https://github.com/seu_usuario/seu_repositorio.git
cd seu_repositorio
```
2. **Crie um ambiente virtual:** Crie e ative um ambiente virtual para gerenciar as dependências do projeto.
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

4. **Instale as dependências:** Use o arquivo `requirements.txt` para instalar todas as dependências necessárias.
```bash
pip install -r requirements.txt
```
## Uso

1. **Prepare seu dataset:**
   - Organize as imagens e vídeos em pastas nomeadas conforme o identificador da pessoa (e.g., `Person1`, `Person2`, etc.).
   - Coloque essas pastas dentro de um diretório chamado `dataset`.

2. **Execute o script principal:** Execute o script `main.py` para iniciar o processo de otimização genética e processamento de dados.
```bash
python main.py
```

## Detalhes do Projeto

Para mais detalhes sobre o projeto, incluindo objetivos, implementação e análise de resultados, acesse a [documentação completa](https://melon-basilisk-d0b.notion.site/Projeto-de-Re-Identifica-o-de-Pessoas-com-Algoritmo-Gen-tico-3332a094d6fa4e84bcdddbe4a8ddc0cf).

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias no projeto.

## Licença

Este projeto está licenciado sob os termos da licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

### Detalhes dos Modelos Utilizados

#### YOLOv8
- **Uso:** Detecção de pessoas em imagens e vídeos.
- **Vantagens:** Alta velocidade e precisão na detecção de objetos.
- **Desvantagens:** Requer recursos computacionais elevados.

#### MobileNetV2
- **Uso:** Extração de características para re-identificação de pessoas.
- **Vantagens:** Modelo leve e eficiente, adequado para dispositivos móveis.
- **Desvantagens:** Menor precisão comparada a modelos maiores.

#### DeepSort
- **Uso:** Rastreamento de pessoas através dos frames.
- **Vantagens:** Mantém a identidade dos objetos rastreados com alta precisão.
- **Desvantagens:** Complexidade na configuração e ajuste.

---

Siga as instruções acima para configurar e executar o projeto. Para qualquer dúvida ou problema, consulte a documentação completa ou entre em contato abrindo uma issue no repositório.
