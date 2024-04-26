# Fast Fourier Transform
* Filtering and compressing on frequency domain.

## Low Pass Filter
A aplicação do filtro passa-baixa em uma imagem no domínio de frequência está no arquivo low_pass.py. Esse arquivo recebe como parâmetro de linha de comando uma imagem. A execução desse arquivo abre uma janela interativa em que é possível escolher o tamanho do raio do filtro, assim como visualizar a imagem original, a imagem filtrada, o filtro utilizado e a magnitude do espectro de frequência. Para fechar a janela é preciso apertar a tecla ‘Q’. O tamanho do raio pode ir de 0 até a metade da altura da imagem.

```
python3 low_pass.py -i image_path
```

## High Pass Filter
A aplicação do filtro passa-alta em uma imagem no domínio de frequência está no arquivo high_pass.py. O funcionamento é semelhante ao do arquivo low_pass.py.

```
python3 high_pass.py -i image_path
```

## Band Pass Filter
A aplicação do filtro passa-faixa em uma imagem no domínio de frequência está no arquivo band_pass.py. O funcionamento é semelhante ao do arquivo low_pass.py, explicado na seção anterior. Adicionalmente, podemos escolher o tamanho dos dois raios de forma interativa.

```
python3 band_pass.py -i image_path
```

## Band Stop Filter
A aplicação do filtro rejeita-faixa em uma imagem no domínio de frequência está no arquivo band_stop.py. O funcionamento é semelhante ao do arquivo band_pass.py, explicado na seção anterior.

```
python3 band_stop.py -i image_path
```

## Compression
A compressão da imagem é feita no arquivo compression.py. Esse arquivo recebe como parâmetros de linha de comando uma imagem e um limiar ou uma porcentagem. Caso sejam passados o limiar e a porcentagem, apenas o limiar será considerado para compressão. A imagem comprimida é salva na pasta compressed_images e possui mesmo nome da imagem original, acrescido do limiar utilizado para compressão, por exemplo: baboon_4490.png.

```
python3 compression.py -i image_path {-t 1000 or -p 50}
```