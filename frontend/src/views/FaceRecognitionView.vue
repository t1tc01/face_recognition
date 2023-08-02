<template>
    <div>
        <h1>{{ title }}</h1>
        <a-button type="primary" ghost @click="onClickCheckIn">
           Check-in
        </a-button>
        <a-button type="primary" ghost @click="onClickCheckOut">
            Check-out
        </a-button>
        <router-view></router-view>
    </div>
</template>

<script>
import OpenCamCheckin from '../components/camera/OpenCamCheckin'
import OpenCamCheckout from '../components/camera/OpenCamCheckout'

import { ref} from 'vue'

export default {
    data() {
        return {
            title: ref('Chọn chế độ mà bạn muốn')
        }
    },
    components: {
        OpenCamCheckin, OpenCamCheckout
    },
    methods: {
        onClickCheckIn() {
            console.log('Open camera for check in!')
            this.$router.push('/check-in-camera')
        },
        onClickCheckOut() { 
            console.log('Open camera for check out!')
            this.$router.push('/check-out-camera')
        }
    }
}
</script>

<style global>
    a-button {
        padding: 0.5cm;
    }
</style>