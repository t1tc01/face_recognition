<template>
    <div>
        <form action="">
            <label for="">Tên người mới</label>
            <input type="text" required> 
    
            <label for="">ID</label>
            <input type="text" required>
        </form>
        <a-button type="primary" ghost @click="addPerson">Đăng ký</a-button>
    </div>
</template>

<script>
export default {
    data() {
        return {
            id: '',
            name: '',
        }
    },
    methods: {
        addPerson() {
            console.log('Add Person')
        }
    }
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