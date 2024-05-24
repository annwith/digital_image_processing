# Document Align

O alinhamento automático de documentos é implementado no arquivo document_align.py. A execução desse arquivo deve ser feita da seguinte forma:

python3 document_align.py -i image_path.png -p 1 -m 0

Os parâmetros passados por linha de comando são o arquivo da imagem que será alinhada, a precisão e o modo do alinhamento (0 para projeção horizontal e 1 para transformada de Hough). A precisão informada define a distância entre os ângulos testados e pode assumir valores como 1, 0.1, 0.01 e assim por diante (qualquer potência de 10 entre 0 e 1). Se um valor inválido for informado, um erro será acusado.

São salvas a imagem alinhada e o texto contido na imagem nas pastas output_images e output_texts. O nome dos arquivos é definido incluindo o nome da imagem, o modo e a inclinação, por exemplo neg_4_mode_0_rotated_356.png.

