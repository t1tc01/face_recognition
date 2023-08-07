<template>
  <div>
    <h1>Thêm người mới vào cơ sở dữ liệu</h1>
    <img class="camera" :src="source" alt="">
  </div>
</template>

<script>
export default {
 
 data() {
   return {
     source: "http://192.168.0.74:8008/streaming_data/"
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