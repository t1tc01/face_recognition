<template>
    <div>
        <form action="">
            <label for="">Tên người mới</label>
            <input type="text" required v-model="name"> 
    
            <label for="">ID</label>
            <input type="text" required v-model="id">
        </form>
        <a-button type="primary" ghost @click="addPerson(id, name)">Đăng ký</a-button>
        <router-view></router-view>

    </div>
</template>

<script>

import router from '@/router'
import {ref} from 'vue'

export default {
    setup() {
        let id = ref('')
        let name = ref('')

        function addPerson(id, name) {
            const axios = require('axios')
            const now = new Date();

            if (id === '' && name === '') {
                console.log('Invalid name and id')
            } else {
                console.log("ID:" + id + " Name: " + name + " Time: " + now.toLocaleString())

                axios.post('http://localhost:8008/send_info-person', {
                    id: id,
                    name: name,
                    add_time: now.toLocaleString()
                })
                .then(function (response) {
                    console.log(response.data);
                    router.push('/add-person-camera')
                })
                .catch(function (error) {
                    console.log(error);
                });
            }
        }

        return {id, name, addPerson}
    },
}
</script>

<style>
div {
    text-align: center;
}
form {
    max-width: 420px;
    margin: 30px auto;
    background: white;
    text-align: left;
    padding: 40px;
    border-radius: 10px;
}
label {
    color: #060606;
    display: inline-block;
    margin: 25px 0 15px;
    font-size: 1em;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: bold;
}
input, select {
    display: block;
    padding: 10px 6px;
    width: 100%;
    box-sizing: border-box;
    border: none;
    border-bottom: 1px solid #ddd;
    color: #555;
}
input[type="checkbox"] {
    display: inline-block;
    width: 16px;
    margin: 0 10px 0 0;
    position: relative;
    top: 2px
}
</style>