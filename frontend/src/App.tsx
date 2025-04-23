import { Routes, Route } from 'react-router-dom'
import { Box } from '@chakra-ui/react'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import TrainModel from './pages/TrainModel'
import TestModel from './pages/TestModel'
import UploadModel from './pages/UploadModel'
import TrainingPage from './pages/TrainingPage'
import ModelsPage from './pages/ModelsPage'

function App() {
  return (
    <Box minH="100vh">
      <Navbar />
      <Box as="main" p={4} maxW="1200px" mx="auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/train" element={<TrainModel />} />
          <Route path="/test" element={<TestModel />} />
          <Route path="/upload" element={<UploadModel />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/models" element={<ModelsPage />} />
        </Routes>
      </Box>
    </Box>
  )
}

export default App 