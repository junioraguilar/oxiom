import { useState } from 'react'
import { 
  Box, 
  Heading, 
  Text, 
  Button, 
  VStack, 
  useToast, 
  Input,
  FormControl,
  FormLabel,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react'
import api from '../api/axios'

const UploadModel = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [modelName, setModelName] = useState('')
  
  const toast = useToast()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]
      setSelectedFile(file)
      // Set default model name from file name (without extension)
      const fileName = file.name
      const modelNameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.')) || fileName
      setModelName(modelNameWithoutExt)
      setUploadSuccess(false)
      setUploadError('')
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      toast({
        title: 'No file selected',
        description: 'Please select a model file to upload',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    // Create form data
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('name', modelName || 'Unnamed model')

    setIsUploading(true)
    setUploadProgress(0)
    setUploadSuccess(false)
    setUploadError('')

    try {
      // Upload model with progress tracking
      const response = await api.post('/api/upload-model', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          )
          setUploadProgress(percentCompleted)
        },
      })

      setUploadSuccess(true)
      toast({
        title: 'Upload successful',
        description: `Model ${selectedFile.name} has been uploaded.`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      })
    } catch (error: any) {
      console.error('Upload failed:', error)
      
      let errorMsg = 'Failed to upload model'
      if (error.response) {
        errorMsg = error.response.data?.error || errorMsg
      }
      
      setUploadError(errorMsg)
      toast({
        title: 'Upload failed',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <Box py={8}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl">
          Upload Your YOLO Model
        </Heading>
        
        <Text>
          Upload your pre-trained YOLO model (.pt file) to use for object detection.
        </Text>
        
        <Box p={6} borderWidth="1px" borderRadius="lg" boxShadow="sm">
          <VStack spacing={4} align="stretch">
            <FormControl isRequired mb={4}>
              <FormLabel>Model Name</FormLabel>
              <Input
                placeholder="Enter a name for your model"
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                isDisabled={isUploading}
              />
            </FormControl>
            <FormControl isRequired mb={4}>
              <FormLabel>Model File (.pt)</FormLabel>
              <Input type="file" accept=".pt" onChange={handleFileChange} isDisabled={isUploading} />
            </FormControl>
            
            {selectedFile && (
              <Text fontSize="sm">
                Selected file: <strong>{selectedFile.name}</strong> ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </Text>
            )}
            
            {isUploading && (
              <Box>
                <Text mb={2}>Uploading... {uploadProgress}%</Text>
                <Progress value={uploadProgress} size="sm" colorScheme="blue" borderRadius="md" />
              </Box>
            )}
            
            {uploadSuccess && (
              <Alert status="success" borderRadius="md">
                <AlertIcon />
                <AlertTitle>Upload Successful!</AlertTitle>
                <AlertDescription>Your model has been uploaded and is ready to use.</AlertDescription>
              </Alert>
            )}
            
            {uploadError && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                <AlertTitle>Upload Failed</AlertTitle>
                <AlertDescription>{uploadError}</AlertDescription>
              </Alert>
            )}
            
            <Button
              colorScheme="blue"
              onClick={handleUpload}
              isLoading={isUploading}
              isDisabled={!selectedFile || !modelName || isUploading}
            >
              Upload Model
            </Button>
          </VStack>
        </Box>
      </VStack>
    </Box>
  )
}

export default UploadModel