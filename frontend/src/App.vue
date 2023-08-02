<template>
  <a-layout style="min-height: 100vh">
    <a-layout-sider v-model="collapsed" collapsible>
      <div class="logo" />
      <a-menu v-model="selectedKeys" theme="dark" mode="inline">
        <a-menu-item key="1">
          <pie-chart-outlined />
          <router-link :to="{name: 'home'}">
            Home
          </router-link>
        </a-menu-item>
        <a-menu-item key="2">
          <desktop-outlined />
          <router-link :to="{name: 'about'}">
              Về chúng tôi
          </router-link>
        </a-menu-item>
        <a-sub-menu key="sub2">
          <template #title>
            <span>
              <team-outlined />
              <span>Tính năng</span>
            </span>
          </template>
          <a-menu-item key="6">
            <router-link :to="{name: 'face-recognition'}">
              Face recognition
            </router-link>
          </a-menu-item>
          <a-menu-item key="8">
            <router-link :to="{name: 'add-person'}">
              Thêm người mới
            </router-link>
          </a-menu-item>
        </a-sub-menu>
        <a-sub-menu key="sub1">
          <template #title>
            <span>
              <user-outlined />
              <span>Thành viên</span>
            </span>
          </template>
          <a-menu-item key="3">Tom</a-menu-item>
          <a-menu-item key="4">Bill</a-menu-item>
          <a-menu-item key="5">Alex</a-menu-item>
        </a-sub-menu>
        <a-menu-item key="9">
          <file-outlined />
          <span>Trích xuất dữ liệu</span>
        </a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header style="background: #fff; padding: 0" />
      <a-layout-content style="margin: 0 16px">
        <div :style="{ padding: '24px', background: '#fff', minHeight: '360px' }">
          <router-view></router-view>
        </div>
      </a-layout-content>
      <a-layout-footer style="text-align: center">
        Face recognition © 2023 Created by ACS AI 
      </a-layout-footer>
    </a-layout>
  </a-layout>
</template>

<script>
export default {
  data() {

  }
}
</script>

<style scoped>
#components-layout-demo-side .logo {
  height: 32px;
  margin: 16px;
  background: rgba(255, 255, 255, 0.3);
}

.site-layout .site-layout-background {
  background: #fff;
}
[data-theme='dark'] .site-layout .site-layout-background {
  background: #141414;
}
</style>