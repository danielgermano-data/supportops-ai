# SupportOps AI

**Treinamento de ML, RAG e agente com ferramentas, com evidências de execução.**

Projeto educacional de portfólio de **Daniel Germano**, desenvolvido com assistência de IA. A proposta é aprender e demonstrar o ciclo inteiro: dados públicos, treinamento, avaliação, artefato de inferência, recuperação de procedimentos, uso de um LLM pretreinado e inspeção das chamadas de ferramentas.

[**Abrir demonstração online**](https://danielgermano-data.github.io/supportops-ai/) · [Relatório do modelo](reports/training.json) · [Execuções reais de RAG e agente](reports/live_evaluation.json) · [Guia de estudo](docs/GUIA_DE_ESTUDO.md) · [Competências e lacunas](docs/COMPETENCIAS.md)

> O classificador foi treinado neste projeto. Os LLMs usados pelo RAG e pelo agente são pretreinados: **não houve treinamento nem fine-tuning de LLM**. A base de procedimentos é ilustrativa; nenhuma conta ou operação bancária é acessada.

## O que funciona em cada modo

| Componente | Demonstração online gratuita | Projeto local |
|---|---|---|
| Classificação em 77 intenções | Pesos reais executados no navegador | Mesmos pesos em Python |
| Idioma do classificador | Inglês | Inglês; o agente normaliza a entrada com tradução por LLM |
| Recuperação de documentos | BM25 lexical em 12 procedimentos bilíngues | Embeddings reais do Ollama com `embeddinggemma` |
| Geração de resposta | Rascunho experimental em inglês com WebLLM/WebGPU; exige revisão | Ollama `granite4.1:3b`, em português, com fontes recuperadas |
| Agente com ferramentas | Código e registros auditáveis no repositório | Chamadas reais a `classify_ticket` e `search_knowledge` |
| Custo de API | Nenhuma API paga; download/processamento no dispositivo | Nenhuma API paga; requer hardware, modelos e energia locais |

A página pública é estática. Ela não usa respostas pré-calculadas para fingir inferência, não conecta ao Ollama de quem visita e não hospeda o backend Python. Os registros JSON são execuções anteriores explicitamente identificadas. A geração no navegador depende do hardware e é uma via experimental: verificar IDs de citações não prova que cada frase está fundamentada.

A geração opcional usa `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` via WebLLM. O primeiro carregamento baixa aproximadamente 1 GB e o registro do modelo estima cerca de 1,6 GB de memória de GPU. Classificação e busca funcionam sem carregar esse LLM. [Registro oficial de modelos](https://github.com/mlc-ai/web-llm/blob/v0.2.89/src/config.ts).

O gerador do navegador recebe apenas o primeiro procedimento da busca para reduzir mistura de orientações de assuntos distintos. A seleção de um ID válido não verifica se o texto gerado usa a fonte corretamente; erros de fundamentação continuam possíveis. A referência local é mais completa e tem seus próprios resultados e limites registrados.

## Resultado do treinamento executado

| Medida | Resultado |
|---|---:|
| Exemplos no ajuste | 8.000 |
| Exemplos na validação | 1.999 |
| Exemplos no teste final | 3.073 |
| Intenções | 77 |
| Acurácia no teste | 87,96% |
| Macro-F1 no teste | 0,8800 |
| Acurácia top-3 | 96,55% |
| Baseline majoritária, acurácia | 1,30% |

O script compara C = 0,5, 2 e 8 **somente na validação**. C = 8 foi selecionado. O TF-IDF é ajustado apenas nas 8.000 linhas de treino. O teste original é consultado após essas escolhas. Quatro repetições exatas foram removidas do treino e sete sobreposições exatas entre treino e teste foram excluídas e registradas. Isso não elimina toda possível semelhança semântica entre exemplos.

O limiar 0,20 foi escolhido na validação para uma meta de 90% de acurácia entre classificações aceitas. No teste, esse limiar aceitou 94,40% dos exemplos e atingiu 91,24% de acurácia entre os aceitos. **Não são probabilidades calibradas, taxas de aprovação em vagas nem garantia de desempenho em produção.**

Arquivos para conferir: [protocolo e métricas](reports/training.json), [índices das partições](reports/splits.json), [predições no teste](reports/test_predictions.csv), [model card](docs/MODEL_CARD.md), [metadados dos pesos](web/model/model.json).

## Executar a demonstração local

Requisitos: Python 3.12 recomendado, Node 22 ou superior para os testes do navegador. Execute os comandos dentro da pasta do projeto.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m uvicorn supportops.api:app --app-dir src --host 127.0.0.1 --port 8765
```

Abra [http://127.0.0.1:8765](http://127.0.0.1:8765). O modelo já está exportado; a classificação e a busca lexical não exigem novo treinamento nem Ollama. A API possui documentação em `/docs`.

Linux/macOS: ative com `source .venv/bin/activate`. O servidor permanece em loopback. Não o exponha à internet como se fosse um serviço de produção autenticado.

## Reproduzir treinamento

```powershell
python scripts/download_data.py
python scripts/train.py
python scripts/export_parity.py
python -m pytest -q
npm test
```

`download_data.py` verifica os checksums do BANKING77 original. Se o fornecedor mudar os arquivos, o script para em vez de aceitar outra versão silenciosamente. `train.py` exporta os pesos em float32, vocabulário e IDF. A inferência não desserializa pickle. Os testes comparam a versão Python e JavaScript nos mesmos exemplos, incluindo decisões de rejeição e pontuações.

Depois que o teste publicado já foi observado, ele não deve virar conjunto de ajuste para novas experiências. Use treino/validação para desenvolver e reserve uma avaliação nova para a próxima comparação final.

## Executar RAG e agente reais

Instale o [Ollama](https://ollama.com/) e execute:

```powershell
ollama pull granite4.1:3b
ollama pull embeddinggemma
python scripts/build_index.py
python scripts/evaluate_live.py
```

Com o servidor local aberto, o botão **Executar agente local com ferramentas** usa o backend. Também há `POST /api/rag` e `POST /api/agent`, com corpo `{"text":"A compra apareceu duas vezes na minha conta. Como devo proceder?"}`.

O agente possui duas ferramentas de leitura, schemas limitados e orçamento de quatro turnos. Não há ferramenta de shell, acesso a contas, estorno, bloqueio ou transferência. A classificação usa uma normalização para inglês produzida por LLM, registrada em `normalized_english_input`; traduções também podem errar. Modelo indisponível resulta em erro explícito. Respostas sem as ferramentas necessárias, com classificação rejeitada ou citações desconhecidas ficam com `needs_review`.

## Evidências e limites do RAG

O [relatório de inferência](reports/live_evaluation.json) registra consultas, documentos recuperados, texto gerado, modelo, latência e chamadas de ferramentas reais. O pequeno conjunto de recuperação foi escrito para esta base; é uma avaliação de desenvolvimento, não um benchmark representativo de atendimento real. Os arquivos `live_evaluation_initial.json` e `live_evaluation_before_translation_fix.json` preservam problemas encontrados durante a implementação.

Na execução registrada de 25/11/2026, o agente concluiu o caso de cartão perdido com ambas as ferramentas e referência válida. No caso de compra duplicada, classificou e recuperou o procedimento correto, mas omitiu a citação: o resultado ficou com `needs_review`. A solicitação para revelar senhas foi recusada e também ficou para revisão. Esses três casos ilustram o comportamento observado; não medem uma taxa geral de segurança ou confiabilidade.

O [teste HTTP local](reports/api_smoke.json) também registra uma chamada real completa por FastAPI, com duas ferramentas e resposta aceita. As [observações no navegador](reports/browser_smoke.json) incluem erros do gerador experimental e saídas retidas. Esse registro distingue execução funcional, conferência de IDs e revisão humana de fundamentação.

Verificar que `[RB-03]` existe no contexto **não verifica automaticamente o significado de uma frase**. Semelhança vetorial também não prova que existe uma resposta. A avaliação humana, o teste de cenários fora do domínio e um conjunto independente de perguntas são próximos passos.

## Estrutura

```text
src/supportops/       classificador, recuperação, Ollama, agente e FastAPI
scripts/             download, treino, índice, avaliação e integridade
data/knowledge.json  procedimentos bilíngues autorais demonstrativos
reports/             métricas, partições, predições e execuções reais
web/                 interface pública e artefato de inferência
tests/               testes Python, JavaScript e paridade de inferência
specs/               arquitetura, decisões e revisão de segurança
docs/                model card e roteiro de estudo
```

## Próximas experiências para aprender

1. Analisar confusões entre intenções parecidas e criar novos casos de avaliação.
2. Medir calibração e falsas aceitações fora do domínio.
3. Comparar BM25 e embeddings em um conjunto independente.
4. Avaliar tradução, fidelidade das citações e uso de ferramentas separadamente.
5. Comparar o baseline com um encoder supervisionado, sem reaproveitar o teste para escolher o modelo.

Esses itens são próximos passos, não resultados já obtidos. O roteiro de sessões está em [Guia de estudo](docs/GUIA_DE_ESTUDO.md).

## Atualizar a página pública

O GitHub Pages usa a raiz da branch `gh-pages`, gerada a partir de `web/`. A CI em `main` verifica o código e os artefatos. Depois de revisar e commitar uma alteração, publique o conteúdo estático:

```powershell
git push origin main
pwsh -File scripts/publish_pages.ps1
```

Em Linux/macOS, o equivalente é `git subtree split --prefix=web HEAD` e um push do SHA retornado para `refs/heads/gh-pages`. A configuração do repositório deve indicar publicação a partir dessa branch, pasta `/`. Confira o workflow automático do Pages e a URL: um push concluído, sozinho, não prova que a implantação terminou.

## Fontes e licença

- BANKING77: [PolyAI dataset card](https://huggingface.co/datasets/PolyAI/banking77), licença CC BY 4.0. Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), 2020. O projeto usa os splits e preserva atribuição; os CSVs são baixados, não vendidos como dados próprios.
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling) e [embeddings](https://docs.ollama.com/api/embed).
- [WebLLM](https://github.com/mlc-ai/web-llm), dependência externa para execução opcional no navegador.
- Código e procedimentos autorais: MIT. Fonte Manrope: SIL OFL, preservada em `web/fonts/OFL.txt`. Modelos de terceiros mantêm suas próprias licenças.

O projeto usa somente o dataset público atribuído acima e procedimentos fictícios autorais.
