<template>
  <div>
    <h1>Chế độ: Check-in</h1>
    <img class="camera" :src="source" alt="">
  </div>
</template>

<script>
import { onMounted, ref } from 'vue'

export default {
 
  data() {
    return {
      source: "http://192.168.0.74:8008/streaming_checkin/"
    }
  },

  beforeRouteLeave(to, from, next) {
    this.isDisabled = true
    this.source = ""
    console.log('router leaved, camera is disable')
    next();
  }
}
</script>

<style>

</style>