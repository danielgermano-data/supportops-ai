# Aprender engenharia de IA com o SupportOps

Este roteiro usa o projeto que já está no repositório. Concluir o roteiro significa conseguir executar, explicar e alterar o trabalho; não basta repetir os números do README.

## Encontro 1 Entender os dados
Execute `python scripts/download_data.py`. Abra `data/dataset_manifest.json` e confira a origem, a licença e os hashes. Leia as primeiras linhas dos CSVs localmente. Explique o que é uma intenção e por que aqui há 77 classes. Compare 3 exemplos de classes parecidas. Pergunta de entrevista: por que o modelo não é automaticamente válido em português?

## Encontro 2 Treinar e comparar
Execute `python scripts/train.py`. Acompanhe cada bloco: normalização, duplicatas, separação, TF-IDF, regressão logística e escolha de C. Leia `reports/training.json`. Compare a baseline majoritária com o modelo e identifique uma classe com F1 baixo. Não escolha hiperparâmetro olhando o teste. Exercício: experimentar outra representação apenas com treino e validação; use um novo conjunto final para decidir uma próxima versão depois que o teste original já foi observado.

## Encontro 3 Entender a abstenção
O limiar foi escolhido na validação para uma meta de acurácia entre respostas aceitas. Não equivale a probabilidade calibrada nem detecta todo texto fora do domínio. Compare `top`, `accepted`, cobertura e acurácia seletiva. Explique o custo de rejeitar mais chamados. Exercício: criar um conjunto separado de pedidos fora do domínio e medir falsas aceitações sem ajustar no teste publicado.

## Encontro 4 Busca e RAG
Leia os 12 procedimentos em `data/knowledge.json`. A versão online usa BM25 lexical. A local usa embeddings `embeddinggemma`. Execute `python scripts/build_index.py` e explique vetor, similaridade e chunk. Abra as consultas de recuperação em `scripts/evaluate_live.py` e compare os resultados. O teste pequeno foi criado manualmente; não representa todos os clientes ou produtos.

## Encontro 5 Ferramentas e agente
Leia `src/supportops/agent.py`. Identifique o prompt, os schemas, o limite de iterações, a tradução para inglês antes da classificação e as mensagens de retorno das ferramentas. O LLM pretreinado decide chamadas; o classificador foi treinado neste projeto. Pergunta de entrevista: qual a diferença entre treinar o classificador, usar um LLM, RAG e fazer fine-tuning? Aqui não fizemos fine-tuning do LLM.

## Encontro 6 Evidência e falhas
Execute `python scripts/evaluate_live.py` com Ollama ativo. Compare `answer`, fontes, `trace`, latência e validação de citações. Uma citação existente não prova que a frase está correta. Leia as falhas e abstenções antes de apresentar o projeto. Acrescente um exemplo adversarial e documente o comportamento observado sem generalizar que o sistema é invulnerável.

## Encontro 7 API testes e publicação
Inicie o FastAPI conforme README, abra `/docs` e teste `/api/classify`. Entenda validação de entrada, origem, limite de chamadas e serialização da inferência. Rode `python -m pytest -q` e `npm test`. Compare o cálculo Python com o JavaScript usando o mesmo arquivo de pesos. Explique por que a demo pública estática não oferece o backend Python online.

## Antes de colocar no currículo
Você deve conseguir demonstrar o pipeline, interpretar as métricas e explicar pelo menos um erro. Descreva a assistência de IA com transparência. Não diga que treinou um LLM, implantou em um banco ou atendeu clientes reais. Exemplo de descrição, após dominar o projeto: “Projeto educacional de classificação de intenções em BANKING77 com TF-IDF e regressão logística, avaliação em teste separado e exportação para inferência no navegador; implementação de RAG e agente local com ferramentas e evidências de execução.”
