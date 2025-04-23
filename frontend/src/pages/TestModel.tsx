import { useState, useEffect } from 'react'
import {
  Box,
  Heading,
  Text,
  VStack,
  Select,
  Button,
  Image,
  HStack,
  Spinner,
  Alert,
  AlertIcon,
  useToast,
  Flex,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb
} from '@chakra-ui/react'
import axios from 'axios'
import DetectionBox from '../components/DetectionBox'

interface Model {
  id: string
  name: string
  path: string
  size: number
  classes?: string[]
}

interface Detection {
  class_id: number
  class_name: string
  confidence: number
  box: {
    x1: number
    y1: number
    x2: number
    y2: number
    width: number
    height: number
  }
}

interface ImageDimensions {
  width: number
  height: number
}

const TestModel = () => {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>('')
  const [resultImageUrl, setResultImageUrl] = useState<string>('')
  const [detections, setDetections] = useState<Detection[]>([])
  const [imageDimensions, setImageDimensions] = useState<ImageDimensions | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25)
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [error, setError] = useState('')
  
  const toast = useToast()

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true)
      try {
        const response = await axios.get('/api/models')
        setModels(response.data.models || [])
      } catch (error) {
        console.error('Error fetching models:', error)
        setError('Failed to load models')
        toast({
          title: 'Error',
          description: 'Failed to load available models',
          status: 'error',
          duration: 5000,
          isClosable: true,
        })
      } finally {
        setIsLoadingModels(false)
      }
    }
    
    fetchModels()
  }, [toast])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]
      setSelectedFile(file)
      
      // Create preview URL
      const fileUrl = URL.createObjectURL(file)
      setPreviewUrl(fileUrl)
      
      // Reset results
      setResultImageUrl('')
      setDetections([])
      setError('')
    }
  }

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(e.target.value)
  }

  const handleDetect = async () => {
    if (!selectedFile || !selectedModel) {
      toast({
        title: 'Missing information',
        description: 'Please select both a model and an image',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setIsLoading(true)
    setError('')
    setDetections([])
    setResultImageUrl('')

    // Create form data for the file upload and detection
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('model_id', selectedModel)
    formData.append('confidence', confidenceThreshold.toString())

    try {
      // Upload image and run detection in a single request
      const detectResponse = await axios.post('/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Set results
      setDetections(detectResponse.data.detections || [])
      setResultImageUrl(detectResponse.data.image_path)
      setImageDimensions(detectResponse.data.image_dimensions)

      toast({
        title: 'Detection successful',
        description: `Found ${detectResponse.data.detections.length} objects`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      })
    } catch (error) {
      console.error('Detection failed:', error)
      
      let errorMsg = 'Detection failed'
      if (axios.isAxiosError(error) && error.response) {
        errorMsg = error.response.data.error || errorMsg
      }
      
      setError(errorMsg)
      toast({
        title: 'Detection failed',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Box py={8}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl">
          Test Object Detection
        </Heading>
        
        <Text>
          Upload an image and select a model to test object detection.
        </Text>

        <Box p={6} borderWidth="1px" borderRadius="lg" boxShadow="sm">
          <VStack spacing={6} align="stretch">
            <Box>
              <Text mb={2} fontWeight="medium">1. Select a model</Text>
              <Select 
                placeholder="Select model" 
                value={selectedModel} 
                onChange={handleModelChange}
                isDisabled={isLoadingModels || models.length === 0}
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({(model.size / (1024 * 1024)).toFixed(2)} MB)
                  </option>
                ))}
              </Select>
              {isLoadingModels && (
                <HStack mt={2}>
                  <Spinner size="sm" />
                  <Text fontSize="sm">Loading models...</Text>
                </HStack>
              )}
              {!isLoadingModels && models.length === 0 && (
                <Text fontSize="sm" color="red.500">
                  No models available. Please upload a model first.
                </Text>
              )}
            </Box>
            
            <Box>
              <Text mb={2} fontWeight="medium">2. Upload an image</Text>
              <Button as="label" htmlFor="image-upload" variant="outline" cursor="pointer" width="full">
                Choose Image
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </Button>
              
              {previewUrl && (
                <Box mt={4} maxW="400px">
                  <Text mb={2} fontSize="sm">Preview:</Text>
                  <Image src={previewUrl} alt="Preview" borderRadius="md" />
                </Box>
              )}
            </Box>

            <Box>
              <Text mb={2} fontWeight="medium">3. Confidence threshold: {(confidenceThreshold * 100).toFixed(0)}%</Text>
              <Slider
                min={0.05}
                max={0.95}
                step={0.05}
                value={confidenceThreshold}
                onChange={(val) => setConfidenceThreshold(val)}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
            </Box>
            
            <Button 
              colorScheme="blue" 
              onClick={handleDetect} 
              isLoading={isLoading} 
              loadingText="Detecting" 
              isDisabled={!selectedFile || !selectedModel || isLoading}
            >
              Detect Objects
            </Button>
            
            {error && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                {error}
              </Alert>
            )}
            
            {resultImageUrl && (
              <VStack align="stretch" spacing={4}>
                <Heading size="md">Detection Results</Heading>
                
                <Box position="relative">
                  <Image src={resultImageUrl} alt="Detection Result" borderRadius="md" />
                  
                  {imageDimensions && detections.map((detection, index) => (
                    <DetectionBox
                      key={index}
                      x1={detection.box.x1}
                      y1={detection.box.y1}
                      width={detection.box.width}
                      height={detection.box.height}
                      label={detection.class_name}
                      confidence={detection.confidence}
                      imageWidth={imageDimensions.width}
                      imageHeight={imageDimensions.height}
                    />
                  ))}
                </Box>
                
                <Box>
                  <Heading size="sm" mb={2}>Detected Objects: {detections.length}</Heading>
                  {detections.map((detection, index) => (
                    <Flex 
                      key={index} 
                      p={3} 
                      bg="gray.50" 
                      _dark={{ bg: 'gray.700' }} 
                      borderRadius="md" 
                      mb={2}
                      justify="space-between"
                      align="center"
                    >
                      <HStack>
                        <Box w={3} h={3} bg="brand.500" borderRadius="full" />
                        <Text fontWeight="medium">{detection.class_name}</Text>
                      </HStack>
                      <Text>Confidence: {(detection.confidence * 100).toFixed(2)}%</Text>
                    </Flex>
                  ))}
                </Box>
              </VStack>
            )}
          </VStack>
        </Box>
      </VStack>
    </Box>
  )
}

export default TestModel 