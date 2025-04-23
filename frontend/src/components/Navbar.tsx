import { Link as RouterLink } from 'react-router-dom'
import {
  Box,
  Flex,
  Text,
  Button,
  Stack,
  Link,
  useColorModeValue,
  useColorMode,
} from '@chakra-ui/react'
import { MoonIcon, SunIcon } from '@chakra-ui/icons'

const Navbar = () => {
  const { colorMode, toggleColorMode } = useColorMode()
  const bgColor = useColorModeValue('white', 'gray.800')
  const textColor = useColorModeValue('gray.600', 'white')

  return (
    <Box
      bg={bgColor}
      px={4}
      boxShadow={'sm'}
      position="sticky"
      top={0}
      zIndex={10}
    >
      <Flex h={16} alignItems={'center'} justifyContent={'space-between'}>
        <Box fontWeight="bold" fontSize="xl">
          <Text 
            as={RouterLink} 
            to="/" 
            color="brand.600"
            _hover={{ textDecoration: 'none' }}
          >
            YOLOv8 Trainer
          </Text>
        </Box>

        <Flex alignItems={'center'}>
          <Stack direction={'row'} spacing={4} alignItems={'center'}>
            <Link
              as={RouterLink}
              to="/"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Home
            </Link>
            <Link
              as={RouterLink}
              to="/train"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Train Model
            </Link>
            <Link
              as={RouterLink}
              to="/training"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Training Monitor
            </Link>
            <Link
              as={RouterLink}
              to="/models"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Models
            </Link>
            <Link
              as={RouterLink}
              to="/test"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Test Model
            </Link>
            <Link
              as={RouterLink}
              to="/upload"
              fontSize={'sm'}
              fontWeight={500}
              color={textColor}
              _hover={{
                textDecoration: 'none',
                color: 'brand.500',
              }}
            >
              Upload Model
            </Link>
            <Button onClick={toggleColorMode} size="sm">
              {colorMode === 'light' ? <MoonIcon /> : <SunIcon />}
            </Button>
          </Stack>
        </Flex>
      </Flex>
    </Box>
  )
}

export default Navbar 