<template>
    <div>
        <h1>Chế  độ: Check-out</h1>
        <img class="camera" :src="source" alt="">
    </div>
</template>

<script>
export default {
    data() {
    return {
      source: "http://localhost:8008/streaming_checkout/"
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