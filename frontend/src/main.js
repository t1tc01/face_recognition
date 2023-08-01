import { createApp } from "vue";
import App from "./App.vue";
import Antd from 'ant-design-vue';
// import 'antd/dist/reset.css';

const app = createApp(App).use(Antd);
app.mount("#app");
