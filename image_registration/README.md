(1) converter as imagens coloridas de entrada em imagens de n´ıveis de cinza.
(2) encontrar pontos de interesse e descritores invariantes locais para o par de imagens.
(3) computar distancias (similaridades) entre cada descritor das duas imagens. ˆ
(4) selecionar as melhores correspondˆencias para cada descritor de imagem.
(5) executar a t´ecnica RANSAC (RANdom SAmple Consensus) para estimar a matriz de homografia
(cv2.findHomography).
(6) aplicar uma projec¸ao de perspectiva ( ˜ cv2.warpPerspective) para alinhar as imagens.
(7) unir as imagens alinhadas e criar a imagem panoramica. ˆ
(8) desenhar retas entre pontos correspondentes no par de imagens.