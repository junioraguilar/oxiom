import { Box, Text } from '@chakra-ui/react';

interface DetectionBoxProps {
  x1: number;
  y1: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  color?: string;
  imageWidth: number;
  imageHeight: number;
}

// Gera uma cor baseada no texto
const getColor = (text: string): string => {
  // Hash simples para gerar um número a partir do texto
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Converte o hash para RGB
  const r = (hash & 0xFF0000) >> 16;
  const g = (hash & 0x00FF00) >> 8;
  const b = hash & 0x0000FF;
  
  return `rgba(${r}, ${g}, ${b}, 0.8)`;
};

const DetectionBox: React.FC<DetectionBoxProps> = ({ 
  x1, 
  y1, 
  width, 
  height, 
  label, 
  confidence, 
  color,
  imageWidth,
  imageHeight
}) => {
  // Calcula a posição percentual para que funcione independente do tamanho da imagem
  const left = (x1 / imageWidth) * 100;
  const top = (y1 / imageHeight) * 100;
  const boxWidth = (width / imageWidth) * 100;
  const boxHeight = (height / imageHeight) * 100;
  
  // Usa a cor fornecida ou gera uma baseada no label
  const boxColor = color || getColor(label);
  
  return (
    <Box
      position="absolute"
      left={`${left}%`}
      top={`${top}%`}
      width={`${boxWidth}%`}
      height={`${boxHeight}%`}
      border={`2px solid ${boxColor}`}
      pointerEvents="none"
    >
      <Box
        position="absolute"
        top="-24px"
        left="0"
        backgroundColor={boxColor}
        px={1}
        py={0.5}
        borderRadius="sm"
        fontSize="xs"
        fontWeight="bold"
        color="white"
        whiteSpace="nowrap"
      >
        {label} {confidence ? `${(confidence * 100).toFixed(0)}%` : ''}
      </Box>
    </Box>
  );
};

export default DetectionBox; 