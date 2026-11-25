# Slides

Apresentação no padrão visual do template do PPGEC (`template/Template_PPGEC_2026_1.pptx`). Cada slide
é um módulo em `build/slides/`; o manifesto em `decks/` define quais entram e em que ordem.

```powershell
conda activate rec_pad
cd slides
python -m build.deck decks/apresentacao.json
```

O arquivo sai em `exports/apresentacao.pptx`, com formas editáveis no PowerPoint (inclusive o diagrama
da arquitetura e o diagrama de diferença crítica) e notas do apresentador em cada slide.

## Números

Os slides leem os resultados de `../results/` por meio de `build/dados.py`. Se o experimento for executado
de novo, basta gerar o deck outra vez. A única exceção é a fração em que o Naive Bayes decide sozinho
(slide 04), que vem da execução exploratória e está fixada no módulo, com a fonte indicada.

## Edição

Edições feitas à mão no `.pptx` se perdem se o arquivo for gerado de novo por cima. Para ajustes finos,
copie o arquivo exportado e edite a cópia, ou altere o módulo do slide e gere de novo. O gerador não
verifica transbordo de texto: confira visualmente depois de gerar.

## Identidade visual

Cores, fonte e moldura (barra superior, rótulo, título com marca vermelha, rodapé) vêm de
`build/pptxkit.py`, extraídas do template oficial. Não invente cores: derive das existentes e registre lá.
