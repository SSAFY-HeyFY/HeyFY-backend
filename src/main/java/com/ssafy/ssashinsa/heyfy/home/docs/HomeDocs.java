package com.ssafy.ssashinsa.heyfy.home.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.home.dto.HomeDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "홈 화면 데이터 조회", description = "사용자 학번, 일반 계좌 및 외화 계좌 정보를 조회합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 홈 화면 데이터를 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = HomeDto.class),
                        examples = @ExampleObject(
                                name = "성공 응답 예시",
                                value = """
                                        {
                                          "studentId": "1234567",
                                          "normalAccount": {
                                            "accountNo": "0014555423195469",
                                            "accountName": "한국은행 수시입출금 상품",
                                            "bankName": "한국은행",
                                            "balance": "1",
                                            "currency": "KRW"
                                          },
                                          "foreignAccount": {
                                            "accountNo": "0010475174188665",
                                            "accountName": "한국은행 외화 수시입출금 상품",
                                            "bankName": "한국은행",
                                            "balance": "0.00",
                                            "currency": "USD"
                                          }
                                        }
                                        """
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "유저를 찾을 수 없음",
                                summary = "JWT 토큰의 사용자 정보를 찾을 수 없는 경우",
                                value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"유저를 찾을 수 없습니다.\"}"
                        )
                )
        )
})
public @interface HomeDocs {
}