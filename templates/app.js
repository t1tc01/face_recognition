const app = Vue.createApp({
    data() {

    },
    methods: {
        goCheckin () {
            location.assign("https://www.example.com");
        }
    },
    computed: {

    }
})

app.mount('#app')