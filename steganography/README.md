# Steganography

Algoritmo de esteganografia capaz de ocultar uma mensagem de texto nos bits menos significativos de uma imagem.

## Encode Message

Para codificar uma mensagem em uma imagem, é preciso executar o arquivo encode.py e passar como parâmetros a imagem que será utilizada, a mensagem que será codificada, quais planos de bits devem ser utilizados e onde a imagem gerada deve ser salva. Também é possível passar um parâmetro para forçar a codificação em todos os planos de bits, se necessário. A execução deve seguir o formato a seguir:

```
python3 encode.py -i image_path -m message_path -b bits_plans -o output_image_path --force
```

Todos os parâmetros, exceto --force, são obrigatórios. O parâmetro bits_plans deve possuir apenas números 0, 1 e/ou 2 separados por vírgula, por exemplo “1” ou “0,1,2”. Outros formatos não serão aceitos e causarão um erro com a seguinte mensagem: Invalid bits plans string. Please use only the numbers 0, 1, or 2 separated by commas and without space.

## Decode Message

Para decodificar uma mensagem em uma imagem, é preciso executar o arquivo decode.py e passar como parâmetros a imagem que será utilizada e quais planos de bits devem ser utilizados. Também é possível passar um parâmetro para forçar a decodificação em todos os planos de bits, se necessário. A execução deve seguir o formato a seguir:

```
python3 decode.py -i image_path -b bits_plans --force
```

Todos os parâmetros, exceto --force, são obrigatórios. O parâmetro bits_plans deve possuir apenas números 0, 1 e/ou 2 separados por vírgula, por exemplo “1” ou “0,1,2”. Outros formatos não serão aceitos e causarão um erro com a seguinte mensagem: Invalid bits plans string. Please use only the numbers 0, 1, or 2 separated by commas and without space.

## Create Message

Arquivo message_generator.py recebe o número de bytes que a mensagem deve ter e gera uma mensagem apenas com caracteres ASCII de 32 a 126, que são os caracteres printáveis. Por conta disso, o bit mais significativo do byte em que o caractere está codificado sempre será zero para textos gerados com esse script. Comando para executar:

```
python3 message_generator -b n_bytes
```

A mensagem gerada é salva na pasta input_messages com formato {n_bytes}_message.txt. Outros arquivos de texto que possam ser codificados para ASCII também são aceitos para codificação e decodificação.


## Compare Messages

A verificação de correspondência entre as mensagens codificadas e decodificadas pode ser feita utilizando o arquivo diff.py. Esse arquivo recebe dois arquivos TXT e compara, retornando as mensagens: “The messages are equal.” ou “The messages are different.”. Para executar o arquivo o comando necessário é: 

```
python3 diff.py -i input_message -o output_message
```

## Show Bit Plans

Para analisar os planos de bits das imagens de entrada e saída, basta executar o arquivo show_bit_plans.py. Esse arquivo recebe duas imagens e quais planos de bits devem ser mostrados. Além disso, o arquivo pode salvar os planos de bits na pasta output_bits_plans caso o parâmetro --save seja passado. Para executar o formato é:

```
python3 show_bit_plans.py -i input_image -o output_image -b bits_plans --save
```

O parâmetro bits_plans pode receber valores de 0 até 7, separados por vírgula e sem espaço.