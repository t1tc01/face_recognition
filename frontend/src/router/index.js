import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import FaceRecognitionView from '@/views/FaceRecognitionView'
import AboutView from '@/views/AboutView'
import AddPersonView from '@/views/AddPersonView'
import OpenCamCheckin from '../components/camera/OpenCamCheckin'
import OpenCamCheckout from '../components/camera/OpenCamCheckout'
import OpenCamAddPerson from '../components/camera/OpenCamAddPerson'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/about',
    name: 'about',
    component: AboutView
  },
  {
    path: '/face-recognition',
    name: 'face-recognition',
    component: FaceRecognitionView
  },
  {
    path: '/add-person',
    name: 'add-person',
    component: AddPersonView
  },
  {
    path:'/check-in-camera',
    name:'check-in-camera',
    component: OpenCamCheckin
  },
  {
    path:'/check-out-camera',
    name:'check-out-camera',
    component: OpenCamCheckout
  },
  {
    path: '/add-person-camera',
    name: 'add-person-camera',
    component: OpenCamAddPerson
  }
]


const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})


export default router
