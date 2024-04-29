# Descrição

O trabalho foi feito em python, utilizando paradigma POO.

Libs utilizadas:

- `glfw` (mesma função do pygame)
- `OpenGL`
- `numpy`
- `PIL` (foi utilizado apenas para ler as texturas)

```bash
pip3 install glfw OpenGL numpy PIL
```

Versão do python utilizada: `3.8.10`

## Como executar

Para executar, basta rodar o arquivo `Game.py` com o python (dentro da pasta `src`):

```bash
cd src
python3 Game.py
```

## Demo

Link com um video curto mostrando o trabalho

<https://drive.google.com/file/d/1K3coTR5NKAK5TYF3DDVnD5Ao5sucJ59Q/view?usp=sharing>

## Sobre o jogo

A ideia do jogo é ser um simulador de barman, onde o jogador recebe pedidos e deve atender os pedidos e montar as bebidas corretamente, similar ao Overcooked.

Infelizmente paramos no algoritmo de detecção de colisão do clique do mouse, não conseguimos resolver os bugs a tempo.

## Classes

- Entity
  - Classe abstrata que representa uma entidade na cena, possui um array com suas coordenadas e um array com os angulos de rotaçao
  - Entidades (classes filho):
    - Camera
    - BarCounter (balcao do bar)
    - Beer (garrafa de cerveja)
    - Whisky (garrafa de whisky)
    - Fan (ventilador)
    - Walls (paredes)
    - Floors (chão que também é usado como teto)
    - Lamp (lampada suspensa)
    - Light (entidade de luz que utiliza o modelo de reflexão de phong)

- Game (classe principal que inicializa o jogo)

- GraphicsEngine (classe responsavel por desenhar na tela e se comunicar com o opengl)

- Mesh (classe de malha que le os arquivos .obj)

- Scene (classe que centraliza as instancias das entidades e organiza a cena)

- Shader (classe que é responsavel pelos arquivos glsl de shaders)

- Texture (classe que importa a imagem de textura e aplica a textura no opengl)
