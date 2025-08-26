package com.ssafy.ssashinsa.heyfy.shinhanApi.client;

import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.member.ShinhanUserResponseDto;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class ShinhanMemberApiClientTest {

    @Autowired
    ShinhanMemberApiClient shinhanMemberApiClient;

    @Test
    @DisplayName("신한 멤버 등록 API 실행 테스트")
    void registerMemberTest() {
        String email = "dbtkd1801@example.com";
        try {
            ShinhanUserResponseDto result = shinhanMemberApiClient.registerMember(email);
            System.out.println("registerMember result: " + result);
        } catch (Exception e) {
            System.out.println("registerMember error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("신한 멤버 조회 API 실행 테스트")
    void getMemberTest() {
        String email = "dbtkd1801@example.com";
        try {
            ShinhanUserResponseDto result = shinhanMemberApiClient.getMember(email);
            System.out.println("getMember result: " + result);
        } catch (Exception e) {
            System.out.println("getMember error: " + e.getMessage());
        }
    }

}
