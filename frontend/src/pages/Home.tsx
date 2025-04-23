import { Box, Heading, Text, Button, SimpleGrid, Icon, VStack, useColorModeValue } from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'
import { FaRobot, FaUpload, FaFlask, FaList } from 'react-icons/fa'
import { IconType } from 'react-icons'

interface FeatureCardProps {
  title: string;
  icon: IconType;
  description: string;
  linkTo: string;
}

const FeatureCard = ({ title, icon, description, linkTo }: FeatureCardProps) => {
  const bg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  return (
    <Box
      p={6}
      bg={bg}
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="lg"
      boxShadow="sm"
      transition="all 0.3s"
      _hover={{ transform: 'translateY(-5px)', boxShadow: 'md' }}
    >
      <VStack spacing={4} align="flex-start">
        <Icon as={icon} boxSize={10} color="brand.500" />
        <Heading size="md">{title}</Heading>
        <Text color="gray.500">{description}</Text>
        <Button as={RouterLink} to={linkTo} colorScheme="blue" variant="outline">
          Get Started
        </Button>
      </VStack>
    </Box>
  )
}

const Home = () => {
  return (
    <Box py={10}>
      <Box textAlign="center" mb={12}>
        <Heading as="h1" size="2xl" mb={3}>
          YOLOv8 Model Trainer
        </Heading>
        <Text fontSize="xl" maxW="2xl" mx="auto" color="gray.500">
          Train, test, and upload YOLOv8 models for object detection with an easy-to-use interface
        </Text>
      </Box>

      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={10} px={4}>
        <FeatureCard
          title="Train Models"
          icon={FaRobot}
          description="Upload your dataset and train custom YOLOv8 models to detect objects in images and videos."
          linkTo="/train"
        />
        <FeatureCard
          title="Test Model"
          icon={FaFlask}
          description="Test your trained models by uploading images and visualizing the detected objects with bounding boxes."
          linkTo="/test"
        />
        <FeatureCard
          title="Upload Models"
          icon={FaUpload}
          description="Upload pre-trained YOLO models to use in your object detection projects."
          linkTo="/upload"
        />
        <FeatureCard
          title="Models Trained"
          icon={FaList}
          description="View and manage all your trained YOLO models."
          linkTo="/models"
        />
      </SimpleGrid>
    </Box>
  )
}

export default Home 