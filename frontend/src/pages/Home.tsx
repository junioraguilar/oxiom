import { useState } from 'react'
import { Box, Heading, Text, Button, SimpleGrid, Icon, VStack, useColorModeValue, Spacer, Flex, Image } from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'
import { FaRobot, FaUpload, FaFlask, FaList, FaDatabase, FaTable } from 'react-icons/fa'
import { IconType } from 'react-icons'
import DatasetUploadModal from '../components/DatasetUploadModal'

interface FeatureCardProps {
  title: string;
  icon: IconType;
  description: string;
  linkTo: string;
  onClick?: () => void;
}

const FeatureCard = ({ title, icon, description, linkTo, onClick }: FeatureCardProps) => {
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
      display="flex"
      flexDirection="column"
      height="100%"
    >
      <VStack spacing={4} align="flex-start" flex="1">
        <Icon as={icon} boxSize={10} color="brand.500" />
        <Heading size="md">{title}</Heading>
        <Text color="gray.500">{description}</Text>
        <Spacer />
      </VStack>
      <Flex justifyContent="center" width="100%" mt={4}>
        {onClick ? (
          <Button onClick={onClick} colorScheme="blue" variant="outline">
            Get Started
          </Button>
        ) : (
        <Button as={RouterLink} to={linkTo} colorScheme="blue" variant="outline">
          Get Started
        </Button>
        )}
      </Flex>
    </Box>
  )
}

const Home = () => {
  const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);

  const handleOpenDatasetModal = () => {
    setIsDatasetModalOpen(true);
  };

  const handleCloseDatasetModal = () => {
    setIsDatasetModalOpen(false);
  };

  const handleUploadComplete = () => {
    // Could add additional actions here if needed after upload completes
  };

  return (
    <Box py={10}>
      <Box textAlign="center" mb={12}>
        <Heading as="h1" size="2xl" mb={3}>
          <Image src="/logo.png" alt="Logo" boxSize="300px" mx="auto" />
        </Heading>
        <Text fontSize="xl" maxW="2xl" mx="auto" color="gray.500">
          Train, test, and upload models for object detection with an easy-to-use interface
        </Text>
      </Box>

      <SimpleGrid columns={{ base: 1, md: 6 }} spacing={10} px={4}>
        <FeatureCard
          title="Upload Dataset"
          icon={FaDatabase}
          description="Upload your dataset in ZIP format to use for training object detection models."
          linkTo=""
          onClick={handleOpenDatasetModal}
        />
        <FeatureCard
          title="Train Models"
          icon={FaRobot}
          description="Upload your dataset and train custom object detection models to detect objects in images and videos."
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
          description="Upload pre-trained object detection models to use in your projects."
          linkTo="/upload"
        />
        <FeatureCard
          title="Models Trained"
          icon={FaList}
          description="View and manage all your trained object detection models."
          linkTo="/models"
        />
        <FeatureCard
          title="Datasets"
          icon={FaTable}
          description="View and manage all uploaded datasets."
          linkTo="/datasets"
        />
      </SimpleGrid>

      <DatasetUploadModal 
        isOpen={isDatasetModalOpen} 
        onClose={handleCloseDatasetModal} 
        onUploadComplete={handleUploadComplete} 
      />
    </Box>
  )
}

export default Home 