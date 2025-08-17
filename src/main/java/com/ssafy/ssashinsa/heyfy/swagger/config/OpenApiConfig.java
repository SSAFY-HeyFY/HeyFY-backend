package com.ssafy.ssashinsa.heyfy.swagger.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.examples.Example;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

import java.util.List;

@Configuration
public class OpenApiConfig {
    final String securitySchemeName = "bearerAuth";
    final String apiTitle = String.format("%s API", StringUtils.capitalize("HeyFY"));
    Info info = new Info()
            .version("v0.1.0")
            .title("HeyFY")
            .description("토큰 입력 시 Bearer 꼭 추가!!");
    @Bean
    public OpenAPI openAPI() {
        Components components = new Components()
                .addExamples("MissingRequired", new Example()
                        .summary("필수 정보 누락")
                        .value("{\"code\":400,\"message\":\"Api request body invalid\"}"))
                .addExamples("NotParticipant", new Example()
                        .summary("참여자가 아닌 경우")
                        .value("{\"code\":400,\"message\":\"User is not participant\"}"))
                .addExamples("Unauthorized", new Example()
                        .summary("인증 실패")
                        .value("{\"code\":401,\"message\":\"Unauthorized\"}"))
                .addExamples("Forbidden", new Example()
                        .summary("인가 실패")
                        .value("{\"code\":403,\"message\":\"Forbidden\"}"))
                .addExamples("NotFound", new Example()
                        .summary("리소스 없음")
                        .value("{\"code\":404,\"message\":\"Resource not found\"}"))
                .addExamples("Conflict", new Example()
                        .summary("충돌")
                        .value("{\"code\":409,\"message\":\"Conflict\"}"))
                .addExamples("InternalError", new Example()
                        .summary("서버 에러")
                        .value("{\"code\":500,\"message\":\"Internal Server Error\"}"));

        return new OpenAPI()
                .servers(List.of(new Server().url("/").description(apiTitle)))
                .components(components.addSecuritySchemes(securitySchemeName,
                        new SecurityScheme()
                                .name(securitySchemeName)
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("JWT")
                ))
                .info(info);
    }
}